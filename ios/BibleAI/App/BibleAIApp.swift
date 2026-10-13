import SwiftUI
import UIKit

@main
struct BibleAIApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var environment = AppEnvironment()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(environment)
                .task {
                    appDelegate.environment = environment
                    await environment.start()
                }
        }
    }
}

/// Receives background URLSession wake-ups for model downloads.
final class AppDelegate: NSObject, UIApplicationDelegate {
    @MainActor var environment: AppEnvironment?

    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        let handler = UncheckedSendable(completionHandler)
        Task { @MainActor in
            self.environment?.modelStore.backgroundCompletionHandler = { handler.value() }
        }
    }
}

/// Wraps a non-Sendable value that is known to be safe to hand over (UIKit completion handlers).
struct UncheckedSendable<Value>: @unchecked Sendable {
    let value: Value
    init(_ value: Value) { self.value = value }
}
