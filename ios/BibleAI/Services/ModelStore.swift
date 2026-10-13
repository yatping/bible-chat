import BibleAICore
import CryptoKit
import Foundation
import Observation
import OSLog

/// Download state of one model file.
enum ModelState: Equatable, Sendable {
    case notDownloaded
    case downloading(progress: Double)
    case verifying
    case ready(URL)
    case failed(String)

    var isReady: Bool {
        if case .ready = self { true } else { false }
    }
}

/// Downloads the GGUF models listed in the bundled `ModelManifest.json` into Application Support/Models/,
/// verifies their sha256 (streamed, CryptoKit) and excludes them from iCloud backup.
///
/// Downloads use a background `URLSession`, so they continue while the app is suspended.
/// This is the only network access the app performs.
@MainActor
@Observable
final class ModelStore {
    static let backgroundSessionID = "com.bibleai.model-download"

    private(set) var manifest: ModelManifest
    private(set) var states: [String: ModelState] = [:]

    @ObservationIgnored private var session: URLSession!
    @ObservationIgnored private let delegate = DownloadDelegate()
    @ObservationIgnored private var tasksByModel: [String: URLSessionDownloadTask] = [:]
    /// Set by the app delegate when iOS relaunches the app for background session events.
    @ObservationIgnored var backgroundCompletionHandler: (@Sendable () -> Void)?
    nonisolated private static let log = Logger(subsystem: "BibleAI", category: "ModelStore")

    let directory: URL

    init(manifest: ModelManifest = ModelStore.bundledManifest(), directory: URL = ModelStore.defaultDirectory) {
        self.manifest = manifest
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var dir = directory
        try? dir.setResourceValues(values)

        delegate.store = self
        let configuration = URLSessionConfiguration.background(withIdentifier: Self.backgroundSessionID)
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        refreshStates()
        reattachRunningTasks()
    }

    nonisolated static var defaultDirectory: URL {
        URL.applicationSupportDirectory.appendingPathComponent("Models", isDirectory: true)
    }

    static func bundledManifest() -> ModelManifest {
        guard let url = Bundle.main.url(forResource: "ModelManifest", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(ModelManifest.self, from: data) else {
            return ModelManifest(version: 1, models: [])
        }
        return manifest
    }

    // MARK: Queries

    func state(of model: ModelManifest.Entry) -> ModelState { states[model.id] ?? .notDownloaded }

    func fileURL(for model: ModelManifest.Entry) -> URL { directory.appendingPathComponent(model.filename) }

    func readyURL(role: ModelManifest.Role) -> URL? {
        guard let model = manifest.model(role: role), case .ready(let url) = state(of: model) else { return nil }
        return url
    }

    /// True when every model in the manifest is downloaded and verified.
    var allReady: Bool { !manifest.models.isEmpty && manifest.models.allSatisfy { state(of: $0).isReady } }

    /// Overall progress across all models (0…1), weighting by size when known.
    var overallProgress: Double {
        guard !manifest.models.isEmpty else { return 0 }
        let parts = manifest.models.map { model -> Double in
            switch state(of: model) {
            case .ready, .verifying: 1
            case .downloading(let p): p
            default: 0
            }
        }
        return parts.reduce(0, +) / Double(parts.count)
    }

    // MARK: Actions

    func downloadAll() {
        for model in manifest.models where !state(of: model).isReady { download(model) }
    }

    func download(_ model: ModelManifest.Entry) {
        if case .downloading = state(of: model) { return }
        let task = session.downloadTask(with: model.url)
        task.taskDescription = model.id
        if model.bytes > 0 { task.countOfBytesClientExpectsToReceive = model.bytes }
        tasksByModel[model.id] = task
        states[model.id] = .downloading(progress: 0)
        task.resume()
    }

    func cancel(_ model: ModelManifest.Entry) {
        tasksByModel.removeValue(forKey: model.id)?.cancel()
        states[model.id] = .notDownloaded
    }

    func delete(_ model: ModelManifest.Entry) {
        cancel(model)
        try? FileManager.default.removeItem(at: fileURL(for: model))
        states[model.id] = .notDownloaded
    }

    /// Recomputes states from the files on disk (size check; the hash is verified once after download).
    func refreshStates() {
        for model in manifest.models {
            let url = fileURL(for: model)
            guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64 else {
                if case .downloading = states[model.id] {} else { states[model.id] = .notDownloaded }
                continue
            }
            states[model.id] = (model.bytes == 0 || size == model.bytes) ? .ready(url) : .failed("Unexpected file size.")
        }
    }

    private func reattachRunningTasks() {
        session.getAllTasks { tasks in
            let running = tasks.compactMap { task -> (String, URLSessionDownloadTask)? in
                guard let download = task as? URLSessionDownloadTask, let id = task.taskDescription else { return nil }
                return (id, download)
            }
            Task { @MainActor in
                for (id, task) in running {
                    self.tasksByModel[id] = task
                    self.states[id] = .downloading(progress: task.progress.fractionCompleted)
                }
            }
        }
    }

    // MARK: Delegate callbacks (main actor)

    fileprivate func progress(modelID: String, written: Int64, expected: Int64) {
        let total = expected > 0 ? expected : (manifest.models.first { $0.id == modelID }?.bytes ?? 0)
        guard total > 0 else { return }
        states[modelID] = .downloading(progress: min(1, Double(written) / Double(total)))
    }

    fileprivate func finished(modelID: String, stagedFile: URL) {
        tasksByModel[modelID] = nil
        guard let model = manifest.models.first(where: { $0.id == modelID }) else { return }
        states[modelID] = .verifying
        let destination = fileURL(for: model)
        Task.detached(priority: .utility) {
            let result = Self.verifyAndInstall(stagedFile, model: model, destination: destination)
            await MainActor.run { self.states[modelID] = result }
        }
    }

    fileprivate func failed(modelID: String, message: String) {
        tasksByModel[modelID] = nil
        states[modelID] = .failed(message)
    }

    fileprivate func backgroundEventsFinished() {
        backgroundCompletionHandler?()
        backgroundCompletionHandler = nil
    }

    // MARK: Verification

    nonisolated static func verifyAndInstall(_ staged: URL, model: ModelManifest.Entry, destination: URL) -> ModelState {
        defer { try? FileManager.default.removeItem(at: staged) }
        do {
            if model.hasChecksum {
                let digest = try sha256(of: staged)
                guard digest == model.sha256.lowercased() else {
                    return .failed("Checksum mismatch – the download is corrupt. Please retry.")
                }
            } else {
                #if DEBUG
                log.warning("manifest has no sha256 for \(model.id, privacy: .public); skipping verification (DEBUG only)")
                #else
                return .failed("The model manifest has no checksum for \(model.filename).")
                #endif
            }
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: staged, to: destination)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var installed = destination
            try installed.setResourceValues(values)
            return .ready(destination)
        } catch {
            return .failed(error.localizedDescription)
        }
    }

    /// Streams the file through SHA-256 in 4 MB chunks (models are GBs; never load them whole).
    nonisolated static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// URLSession delegate; forwards events to the main-actor `ModelStore`.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    weak var store: ModelStore?

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let id = downloadTask.taskDescription else { return }
        Task { @MainActor [weak store] in
            store?.progress(modelID: id, written: totalBytesWritten, expected: totalBytesExpectedToWrite)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let id = downloadTask.taskDescription else { return }
        if let http = downloadTask.response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            Task { @MainActor [weak store] in store?.failed(modelID: id, message: "Server returned HTTP \(http.statusCode).") }
            return
        }
        // The temporary file is deleted when this method returns: move it synchronously.
        let staged = ModelStore.defaultDirectory.appendingPathComponent("\(id).download")
        do {
            try? FileManager.default.removeItem(at: staged)
            try FileManager.default.moveItem(at: location, to: staged)
        } catch {
            let message = error.localizedDescription
            Task { @MainActor [weak store] in store?.failed(modelID: id, message: message) }
            return
        }
        Task { @MainActor [weak store] in store?.finished(modelID: id, stagedFile: staged) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error, let id = task.taskDescription else { return }
        if (error as NSError).code == NSURLErrorCancelled { return }
        let message = error.localizedDescription
        Task { @MainActor [weak store] in store?.failed(modelID: id, message: message) }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor [weak store] in store?.backgroundEventsFinished() }
    }
}
