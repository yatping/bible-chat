import BibleAICore
import SwiftUI

/// Lets the user see, delete and disable what the assistant remembers.
struct MemorySettingsView: View {
    @Environment(AppEnvironment.self) private var env
    @State private var active: [MemoryRecord] = []
    @State private var history: [MemoryRecord] = []
    @State private var confirmDeleteAll = false

    var body: some View {
        @Bindable var env = env
        List {
            Section {
                Toggle("Remember things about me", isOn: $env.memoryEnabled)
            } footer: {
                Text("Memories (like your preferred translation or answer style) are extracted on this device and never leave it.")
            }

            ForEach(MemoryType.allCases, id: \.self) { type in
                let items = active.filter { $0.type == type }
                if !items.isEmpty {
                    Section(type.displayName) {
                        ForEach(items) { MemoryRow(memory: $0) }
                            .onDelete { delete(items, at: $0) }
                    }
                }
            }

            if !history.isEmpty {
                Section("History") {
                    ForEach(history) { MemoryRow(memory: $0).foregroundStyle(.secondary) }
                        .onDelete { delete(history, at: $0) }
                }
            }

            if !active.isEmpty || !history.isEmpty {
                Section {
                    Button("Delete all memories", role: .destructive) { confirmDeleteAll = true }
                }
            }
        }
        .overlay {
            if active.isEmpty && history.isEmpty {
                ContentUnavailableView("No memories yet", systemImage: "brain",
                                       description: Text("Tell the assistant your preferences, e.g. “I prefer short answers.”"))
            }
        }
        .navigationTitle("Memory")
        .confirmationDialog("Delete all memories?", isPresented: $confirmDeleteAll, titleVisibility: .visible) {
            Button("Delete all", role: .destructive) {
                try? env.memories?.deleteAll()
                reload()
            }
        }
        .onAppear(perform: reload)
    }

    private func reload() {
        active = (try? env.memories?.activeMemories()) ?? []
        history = (try? env.memories?.supersededMemories()) ?? []
    }

    private func delete(_ items: [MemoryRecord], at offsets: IndexSet) {
        for index in offsets { try? env.memories?.delete(items[index].id) }
        reload()
    }
}

private struct MemoryRow: View {
    let memory: MemoryRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(memory.content)
            Text(memory.updatedAt, format: .dateTime.day().month().year())
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
