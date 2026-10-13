import BibleAICore
import Foundation

/// Locates and opens the bundled read-only content (`Content/bible.sqlite` + `Content/rag_embeddings.f16`).
enum ContentStore {
    enum Status {
        case ready(ContentDatabase, SemanticSearch?)
        case missing(String)
    }

    static var bundleDirectory: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("Content", isDirectory: true)
    }

    static func open() -> Status {
        guard let directory = bundleDirectory,
              FileManager.default.fileExists(atPath: directory.appendingPathComponent("bible.sqlite").path) else {
            return .missing("The Bible database isn't bundled with this build. Run content/ to build it, then ios/scripts/bundle_content.sh and rebuild the app.")
        }
        do {
            let database = try ContentDatabase(url: directory.appendingPathComponent("bible.sqlite"))
            let embeddings = directory.appendingPathComponent(database.meta["embedding_file"] ?? "rag_embeddings.f16")
            let semantic = FileManager.default.fileExists(atPath: embeddings.path)
                ? try? SemanticSearch.load(database: database, embeddingsURL: embeddings)
                : nil
            return .ready(database, semantic)
        } catch {
            return .missing("The Bible database could not be opened: \(error)")
        }
    }
}
