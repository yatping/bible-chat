import BibleAICore
import SwiftUI

/// All conversations, newest first. Also routes "Ask about this passage" from the Study tab into a new chat.
struct ConversationListView: View {
    @Environment(AppEnvironment.self) private var env
    @State private var conversations: [Conversation] = []
    @State private var path: [Route] = []

    enum Route: Hashable {
        case conversation(id: String, initialQuestion: String?)
    }

    var body: some View {
        NavigationStack(path: $path) {
            List {
                if let message = env.contentMessage {
                    Section {
                        Label(message, systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                ForEach(conversations) { conversation in
                    NavigationLink(value: Route.conversation(id: conversation.id, initialQuestion: nil)) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(conversation.title ?? "New conversation")
                                .font(.body)
                                .lineLimit(2)
                            Text(conversation.updatedAt, format: .relative(presentation: .named))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete(perform: delete)
            }
            .overlay {
                if conversations.isEmpty {
                    ContentUnavailableView("No conversations", systemImage: "text.bubble",
                                           description: Text("Ask a question about the Bible. Everything stays on your device."))
                }
            }
            .navigationTitle("Bible Chat")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("New chat", systemImage: "square.and.pencil") { newConversation(question: nil) }
                }
            }
            .navigationDestination(for: Route.self) { route in
                switch route {
                case .conversation(let id, let question):
                    ChatView(conversationID: id, initialQuestion: question)
                }
            }
            .onAppear(perform: reload)
            .onChange(of: path) { reload() }
            .onChange(of: env.pendingQuestion, initial: true) { _, question in
                guard let question else { return }
                env.pendingQuestion = nil
                newConversation(question: question)
            }
        }
    }

    private func reload() {
        conversations = (try? env.conversations?.conversations()) ?? []
    }

    private func newConversation(question: String?) {
        guard let conversation = try? env.conversations?.createConversation() else { return }
        path = [.conversation(id: conversation.id, initialQuestion: question)]
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets { try? env.conversations?.delete(conversations[index].id) }
        reload()
    }
}
