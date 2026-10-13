import Foundation
import Testing
@testable import BibleAICore

/// The Swift constants are copies of `shared/` files; regenerate with `ios/scripts/sync_shared_texts.py`.
@Suite("Shared texts are in sync with shared/")
struct SharedTextsTests {
    @Test(arguments: [
        ("system_prompt.txt", SharedTexts.systemPrompt),
        ("memory_extraction_prompt.txt", SharedTexts.memoryExtractionPrompt),
        ("summarization_prompt.txt", SharedTexts.summarizationPrompt),
        ("stopwords.txt", SharedTexts.stopwords),
        ("schema_user.sql", SharedTexts.userSchema),
    ])
    func matchesSharedFile(file: String, constant: String) throws {
        #expect(constant == (try Repo.sharedText(file)), "\(file) drifted; run ios/scripts/sync_shared_texts.py")
    }

    @Test func extractionPromptInsertsMessage() {
        let prompt = MemoryExtractor.prompt(for: "  I prefer the WEB.  ")
        #expect(prompt.hasSuffix("USER MESSAGE:\nI prefer the WEB."))
        #expect(!prompt.contains("{{"))
    }

    @Test func modelManifestExampleDecodes() throws {
        let data = try Data(contentsOf: Repo.shared.appendingPathComponent("model_manifest.example.json"))
        let manifest = try JSONDecoder().decode(ModelManifest.self, from: data)
        #expect(manifest.models.count == 2)
        #expect(manifest.model(role: .generator)?.contextLength == 6144)
        #expect(manifest.model(role: .embedder)?.truncateDim == 256)
    }
}
