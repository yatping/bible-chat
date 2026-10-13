import BibleAICore
import SwiftUI

/// One conversation: streamed answers with tappable [S#] citation chips.
struct ChatView: View {
    @Environment(AppEnvironment.self) private var env
    let conversationID: String
    let initialQuestion: String?

    @State private var model: ChatViewModel?
    @State private var draft = ""
    @State private var selectedSource: CitedSource?
    @FocusState private var inputFocused: Bool

    var body: some View {
        Group {
            if let model {
                content(model)
            } else {
                ProgressView()
            }
        }
        .navigationTitle("Chat")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            guard model == nil else { return }
            let viewModel = ChatViewModel(conversationID: conversationID, env: env)
            model = viewModel
            if let initialQuestion, viewModel.items.isEmpty { viewModel.send(initialQuestion) }
        }
        .sheet(item: $selectedSource) { source in
            SourceSheet(citation: source, chunk: model?.chunk(for: source))
        }
    }

    private func content(_ model: ChatViewModel) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(model.items) { item in
                        MessageRow(item: item, liveSources: model.liveSources) { selectedSource = $0 }
                            .id(item.id)
                    }
                    if let error = model.errorMessage {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: model.items.last?.text) {
                if let last = model.items.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
            .safeAreaInset(edge: .bottom) { inputBar(model) }
        }
    }

    private func inputBar(_ model: ChatViewModel) -> some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Ask about the Bible…", text: $draft, axis: .vertical)
                .lineLimit(1...5)
                .focused($inputFocused)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.fill.tertiary, in: RoundedRectangle(cornerRadius: 18))
            if model.isGenerating {
                Button("Stop", systemImage: "stop.circle.fill") { model.stop() }
                    .labelStyle(.iconOnly)
                    .font(.title)
            } else {
                Button("Send", systemImage: "arrow.up.circle.fill") {
                    model.send(draft)
                    draft = ""
                }
                .labelStyle(.iconOnly)
                .font(.title)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }
}

/// A single chat bubble; assistant answers show their cited sources as chips.
private struct MessageRow: View {
    let item: ChatViewModel.Item
    let liveSources: [RetrievedSource]
    let onSelect: (CitedSource) -> Void

    var body: some View {
        if item.role == .user {
            HStack {
                Spacer(minLength: 40)
                Text(item.text)
                    .padding(12)
                    .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 16))
                    .textSelection(.enabled)
            }
        } else {
            VStack(alignment: .leading, spacing: 8) {
                if item.text.isEmpty && item.isStreaming {
                    ProgressView().controlSize(.small)
                } else {
                    Text(Self.markdown(item.text))
                        .textSelection(.enabled)
                }
                let chips = item.isStreaming ? liveSources.map(\.cited) : item.citations
                if !chips.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(chips) { source in
                                Button { onSelect(source) } label: {
                                    Text("\(source.id) · \(source.reference)")
                                        .font(.caption)
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 6)
                                        .background(.fill.secondary, in: Capsule())
                                }
                                .buttonStyle(.plain)
                                .accessibilityLabel("Source \(source.id), \(source.reference)")
                            }
                        }
                    }
                }
            }
        }
    }

    static func markdown(_ text: String) -> AttributedString {
        (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(text)
    }
}

/// Shows a cited source: reference, type and full text.
struct SourceSheet: View {
    let citation: CitedSource
    let chunk: Chunk?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text(citation.reference).font(.title2.bold())
                    Label(citation.type.capitalized, systemImage: citation.type == "scripture" ? "book.closed" : "text.quote")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    if let chunk {
                        if chunk.title != chunk.reference { Text(chunk.title).font(.headline) }
                        Text(chunk.text).textSelection(.enabled)
                        if let translation = chunk.translation {
                            Text(translation == "WEB" ? "World English Bible (public domain)" : translation)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        Text("The source text is not available.").foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
            }
            .navigationTitle(citation.id)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .presentationDetents([.medium, .large])
    }
}
