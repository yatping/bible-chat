import BibleAICore
import Foundation
import Observation
import OSLog

/// Dependency container: owns the databases, engines and the orchestrator.
@MainActor
@Observable
final class AppEnvironment {
    enum Phase: Equatable {
        case needsModels
        case loading
        case ready
        case failed(String)
    }

    enum Tab: Hashable { case chats, study, settings }

    let modelStore: ModelStore
    private(set) var phase: Phase = .loading
    private(set) var content: ContentDatabase?
    private(set) var contentMessage: String?
    private(set) var orchestrator: AIOrchestrator?
    private(set) var userDatabase: UserDatabase?

    var selectedTab: Tab = .chats
    /// A question queued from the Study tab ("Ask about this passage").
    var pendingQuestion: String?

    var memoryEnabled: Bool {
        didSet {
            UserDefaults.standard.set(memoryEnabled, forKey: "memoryEnabled")
            let settings = OrchestratorSettings(memoryEnabled: memoryEnabled)
            Task { await orchestrator?.updateSettings(settings) }
        }
    }

    /// DEBUG: run without downloaded models (scripted answers), e.g. in the simulator.
    var useMockModel: Bool {
        didSet {
            UserDefaults.standard.set(useMockModel, forKey: "useMockModel")
            Task { await start() }
        }
    }

    @ObservationIgnored private var semantic: SemanticSearch?
    nonisolated private static let log = Logger(subsystem: "BibleAI", category: "App")

    init(modelStore: ModelStore? = nil) {
        self.modelStore = modelStore ?? ModelStore()
        UserDefaults.standard.register(defaults: ["memoryEnabled": true])
        memoryEnabled = UserDefaults.standard.bool(forKey: "memoryEnabled")
        #if DEBUG
        useMockModel = UserDefaults.standard.bool(forKey: "useMockModel") || !EngineFactory.hasLlama
        #else
        useMockModel = false
        #endif
    }

    var conversations: ConversationStore? { userDatabase.map(ConversationStore.init) }
    var memories: MemoryStore? { userDatabase.map(MemoryStore.init) }

    /// Opens databases and, if models are available (or mock mode is on), builds the engines.
    func start() async {
        do {
            if userDatabase == nil {
                userDatabase = try UserDatabase(url: URL.applicationSupportDirectory.appendingPathComponent("user.sqlite"))
            }
        } catch {
            phase = .failed("Could not open your conversation database: \(error)")
            return
        }
        if content == nil, contentMessage == nil {
            switch ContentStore.open() {
            case .ready(let database, let semantic):
                content = database
                self.semantic = semantic
            case .missing(let message):
                contentMessage = message
            }
        }

        guard useMockModel || modelStore.allReady else {
            phase = .needsModels
            return
        }
        phase = .loading
        let llm = EngineFactory.makeLLM(modelURL: useMockModel ? nil : modelStore.readyURL(role: .generator))
        let embedder = useMockModel ? nil : EngineFactory.makeEmbedder(modelURL: modelStore.readyURL(role: .embedder))
        do {
            let rag = try content.map { try RAGEngine(database: $0, semantic: embedder == nil ? nil : semantic, embedder: embedder) }
            orchestrator = AIOrchestrator(llm: llm, embedder: embedder, rag: rag, userDatabase: userDatabase!,
                                          settings: OrchestratorSettings(memoryEnabled: memoryEnabled))
            phase = .ready
            // Warm up the model in the background; the first question waits on it anyway.
            Task.detached(priority: .utility) {
                do { try await llm.load() } catch { Self.log.error("model load failed: \(String(describing: error), privacy: .public)") }
            }
        } catch {
            phase = .failed("Could not start the assistant: \(error)")
        }
    }

    /// Called by the setup screen when downloads complete.
    func modelsDidChange() async {
        modelStore.refreshStates()
        if modelStore.allReady, phase == .needsModels { await start() }
    }

    func ask(about question: String) {
        pendingQuestion = question
        selectedTab = .chats
    }
}
