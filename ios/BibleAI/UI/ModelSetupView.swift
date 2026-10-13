import BibleAICore
import SwiftUI

/// First-run screen: downloads the on-device models with progress.
struct ModelSetupView: View {
    @Environment(AppEnvironment.self) private var env

    var body: some View {
        let store = env.modelStore
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Image(systemName: "book.and.wrench")
                            .font(.largeTitle)
                            .foregroundStyle(.tint)
                        Text("Download the AI models")
                            .font(.title2.bold())
                        Text("Bible Chat runs entirely on your iPhone. The models are downloaded once (Wi-Fi recommended); after that nothing leaves your device.")
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 8)
                }

                if store.manifest.models.isEmpty {
                    Section {
                        Label("No model manifest is bundled with this build.", systemImage: "exclamationmark.triangle")
                    }
                }

                Section("Models") {
                    ForEach(store.manifest.models) { model in
                        ModelRow(model: model, state: store.state(of: model))
                    }
                }

                Section {
                    Button("Download all", systemImage: "arrow.down.circle") { store.downloadAll() }
                        .disabled(store.manifest.models.isEmpty || store.allReady)
                    if store.allReady {
                        Button("Continue") { Task { await env.modelsDidChange() } }
                    }
                }

                #if DEBUG
                Section {
                    Button("Use mock model (DEBUG)") { env.useMockModel = true }
                } footer: {
                    Text("Runs the UI with scripted answers, e.g. in the simulator.")
                }
                #endif
            }
            .navigationTitle("Setup")
            .onChange(of: store.allReady) { _, ready in
                if ready { Task { await env.modelsDidChange() } }
            }
        }
    }
}

private struct ModelRow: View {
    @Environment(AppEnvironment.self) private var env
    let model: ModelManifest.Entry
    let state: ModelState

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(model.role == .generator ? "Answer model" : "Search model")
                    .font(.headline)
                Spacer()
                if model.bytes > 0 {
                    Text(Int64(model.bytes), format: .byteCount(style: .file))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text(model.filename).font(.caption).foregroundStyle(.secondary)
            switch state {
            case .notDownloaded:
                Button("Download") { env.modelStore.download(model) }
            case .downloading(let progress):
                ProgressView(value: progress) {
                    Text("Downloading… \(progress, format: .percent.precision(.fractionLength(0)))").font(.caption)
                }
                Button("Cancel", role: .cancel) { env.modelStore.cancel(model) }.font(.caption)
            case .verifying:
                ProgressView { Text("Verifying…").font(.caption) }
            case .ready:
                Label("Ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            case .failed(let message):
                Label(message, systemImage: "xmark.octagon").foregroundStyle(.red).font(.caption)
                Button("Retry") { env.modelStore.download(model) }
            }
        }
        .padding(.vertical, 4)
    }
}
