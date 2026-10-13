import BibleAICore
import SwiftUI

/// Browse book → chapter → verses from the content DB, and ask about a passage.
struct StudyView: View {
    @Environment(AppEnvironment.self) private var env
    @State private var books: [Book] = []

    var body: some View {
        NavigationStack {
            Group {
                if env.content == nil {
                    ContentUnavailableView("Bible not available", systemImage: "book.closed",
                                           description: Text(env.contentMessage ?? "The Bible database is missing."))
                } else {
                    List {
                        ForEach(["OT", "NT"], id: \.self) { testament in
                            Section(testament == "OT" ? "Old Testament" : "New Testament") {
                                ForEach(books.filter { $0.testament == testament }) { book in
                                    NavigationLink(book.name, value: book)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Study")
            .navigationDestination(for: Book.self) { ChapterGrid(book: $0) }
            .navigationDestination(for: ChapterRoute.self) { ChapterView(book: $0.book, chapter: $0.chapter) }
            .task { books = (try? env.content?.books()) ?? [] }
        }
    }
}

struct ChapterRoute: Hashable {
    let book: Book
    let chapter: Int
}

private struct ChapterGrid: View {
    let book: Book

    var body: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 56))], spacing: 12) {
                ForEach(1...max(1, book.chapters), id: \.self) { chapter in
                    NavigationLink(value: ChapterRoute(book: book, chapter: chapter)) {
                        Text("\(chapter)")
                            .font(.body.monospacedDigit())
                            .frame(maxWidth: .infinity, minHeight: 44)
                            .background(.fill.tertiary, in: RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("\(book.name) chapter \(chapter)")
                }
            }
            .padding()
        }
        .navigationTitle(book.name)
    }
}

private struct ChapterView: View {
    @Environment(AppEnvironment.self) private var env
    let book: Book
    let chapter: Int
    @State private var verses: [Verse] = []

    var body: some View {
        List(verses) { verse in
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("\(verse.number)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Text(verse.text)
                    .textSelection(.enabled)
            }
        }
        .listStyle(.plain)
        .overlay {
            if verses.isEmpty { ContentUnavailableView("No verses", systemImage: "text.page") }
        }
        .navigationTitle("\(book.name) \(chapter)")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Ask about this passage", systemImage: "sparkles") {
                    env.ask(about: "Explain \(book.name) \(chapter).")
                }
            }
        }
        .task { verses = (try? env.content?.verses(book: book.index, chapter: chapter)) ?? [] }
    }
}
