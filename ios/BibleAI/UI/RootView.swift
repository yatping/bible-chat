import SwiftUI

/// Top level: model setup until the models are ready, then the main tabs.
struct RootView: View {
    @Environment(AppEnvironment.self) private var env

    var body: some View {
        @Bindable var env = env
        switch env.phase {
        case .needsModels:
            ModelSetupView()
        case .loading:
            ProgressView("Preparing…")
        case .failed(let message):
            ContentUnavailableView("Something went wrong", systemImage: "exclamationmark.triangle", description: Text(message))
        case .ready:
            TabView(selection: $env.selectedTab) {
                ConversationListView()
                    .tabItem { Label("Chats", systemImage: "bubble.left.and.text.bubble.right") }
                    .tag(AppEnvironment.Tab.chats)
                StudyView()
                    .tabItem { Label("Study", systemImage: "book") }
                    .tag(AppEnvironment.Tab.study)
                SettingsView()
                    .tabItem { Label("Settings", systemImage: "gearshape") }
                    .tag(AppEnvironment.Tab.settings)
            }
        }
    }
}
