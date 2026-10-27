import Foundation

enum UnlockResult {
    case ok
    case wrong(triesLeft: Int)
    case throttled(until: Date)
}

/// The parental lock. The password/PIN is stored only as PBKDF2-SHA256(salt, 310k)
/// in the Keychain; it never leaves the island. Unlocks last 5 minutes. Repeated
/// failures back off exponentially, and the back-off survives a restart.
@MainActor
final class GuardianLock: ObservableObject {
    static let account = "guardian-password"
    static let iterations = 310_000
    static let unlockWindow: TimeInterval = 5 * 60
    static let freeTries = 3

    @Published private(set) var hasPassword: Bool
    @Published private(set) var unlockedUntil: Date?
    @Published private(set) var lockedOutUntil: Date?
    private var failures = 0

    private struct Record: Codable {
        var v = 1
        var alg = "pbkdf2-sha256"
        var iterations: Int
        var salt: String
        var hash: String
    }

    private struct ThrottleState: Codable {
        var failures: Int
        var lockedUntil: Double?
    }

    init(probe: Bool = true) {
        hasPassword = probe ? Keychain.read(Self.account) != nil : false
        if probe, let data = try? Data(contentsOf: Paths.lockState),
           let st = try? JSONDecoder().decode(ThrottleState.self, from: data) {
            failures = st.failures
            if let u = st.lockedUntil { lockedOutUntil = Date(timeIntervalSince1970: u) }
        }
    }

    /// Snapshot helper: pretend a password exists without touching the Keychain.
    func fakeForSnapshot(hasPassword: Bool, unlocked: Bool) {
        self.hasPassword = hasPassword
        unlockedUntil = unlocked ? Date().addingTimeInterval(252) : nil
    }

    var isUnlocked: Bool {
        guard let u = unlockedUntil else { return false }
        return u > Date()
    }

    func remainingUnlock(now: Date = Date()) -> TimeInterval {
        guard let u = unlockedUntil else { return 0 }
        return max(0, u.timeIntervalSince(now))
    }

    func throttleRemaining(now: Date = Date()) -> TimeInterval {
        guard let u = lockedOutUntil else { return 0 }
        return max(0, u.timeIntervalSince(now))
    }

    func lockNow() { unlockedUntil = nil }

    /// Sets a new password. Allowed on first run, or while unlocked.
    func setPassword(_ pw: String) -> Bool {
        guard !hasPassword || isUnlocked else { return false }
        let salt = Crypto.randomBytes(16)
        let hash = Crypto.pbkdf2(pw, salt: salt, iterations: Self.iterations)
        let rec = Record(iterations: Self.iterations, salt: salt.base64EncodedString(), hash: hash.base64EncodedString())
        guard let data = try? JSONEncoder().encode(rec), Keychain.write(Self.account, data) else { return false }
        hasPassword = true
        unlockedUntil = Date().addingTimeInterval(Self.unlockWindow)
        resetFailures()
        return true
    }

    func verify(_ pw: String, completion: @escaping (UnlockResult) -> Void) {
        let wait = throttleRemaining()
        if wait > 0 { completion(.throttled(until: lockedOutUntil!)); return }
        guard let data = Keychain.read(Self.account),
              let rec = try? JSONDecoder().decode(Record.self, from: data),
              let salt = Data(base64Encoded: rec.salt), let want = Data(base64Encoded: rec.hash) else {
            completion(.wrong(triesLeft: 0)); return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let got = Crypto.pbkdf2(pw, salt: salt, iterations: rec.iterations)
            let ok = Crypto.constantTimeEqual(got, want)
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    if ok {
                        self.resetFailures()
                        self.unlockedUntil = Date().addingTimeInterval(Self.unlockWindow)
                        completion(.ok)
                    } else {
                        self.failures += 1
                        if self.failures >= Self.freeTries {
                            let secs = min(15 * 60, 30 * pow(2, Double(self.failures - Self.freeTries)))
                            self.lockedOutUntil = Date().addingTimeInterval(secs)
                            self.persist()
                            completion(.throttled(until: self.lockedOutUntil!))
                        } else {
                            self.persist()
                            completion(.wrong(triesLeft: Self.freeTries - self.failures))
                        }
                    }
                }
            }
        }
    }

    private func resetFailures() {
        failures = 0
        lockedOutUntil = nil
        persist()
    }

    private func persist() {
        let st = ThrottleState(failures: failures, lockedUntil: lockedOutUntil?.timeIntervalSince1970)
        if let data = try? JSONEncoder().encode(st) { Paths.writePrivate(data, to: Paths.lockState) }
    }
}
