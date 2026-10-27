import Darwin
import Foundation

/// Unix-socket JSON-lines client for the engine (island-protocol.md v1).
/// Reconnects with exponential back-off (0.5 s ... 8 s). Callbacks arrive on the main queue.
final class EngineClient {
    let path: String
    var onMessage: (([String: Any]) -> Void)?
    var onConnection: ((Bool) -> Void)?

    private let lock = NSLock()
    private var fd: Int32 = -1
    private var running = false
    private var thread: Thread?

    init(path: String) { self.path = path }

    func start() {
        lock.lock(); defer { lock.unlock() }
        guard !running else { return }
        running = true
        let t = Thread { [weak self] in self?.loop() }
        t.name = "ScreenGuardian.engine"
        t.qualityOfService = .utility
        thread = t
        t.start()
    }

    func stop() {
        lock.lock()
        running = false
        let f = fd
        fd = -1
        lock.unlock()
        if f >= 0 { shutdown(f, SHUT_RDWR); close(f) }
    }

    var isRunning: Bool { lock.lock(); defer { lock.unlock() }; return running }

    /// Sends one JSON object. Returns false when not connected.
    @discardableResult
    func send(_ obj: [String: Any]) -> Bool {
        guard var data = try? JSONSerialization.data(withJSONObject: obj) else { return false }
        data.append(0x0A)
        lock.lock()
        let f = fd
        lock.unlock()
        guard f >= 0 else { return false }
        var ok = true
        data.withUnsafeBytes { raw in
            var off = 0
            while off < raw.count {
                let n = Darwin.write(f, raw.baseAddress! + off, raw.count - off)
                if n <= 0 { ok = false; break }
                off += n
            }
        }
        if !ok { shutdown(f, SHUT_RDWR) }
        return ok
    }

    private func loop() {
        var backoff = 0.5
        while isRunning {
            let f = connectSocket()
            if f >= 0 {
                lock.lock(); fd = f; lock.unlock()
                backoff = 0.5
                DispatchQueue.main.async { self.onConnection?(true) }
                readLoop(f)
                lock.lock()
                if fd == f { fd = -1 }
                lock.unlock()
                close(f)
                DispatchQueue.main.async { self.onConnection?(false) }
            }
            guard isRunning else { break }
            Thread.sleep(forTimeInterval: backoff)
            backoff = min(8, backoff * 2)
        }
    }

    private func connectSocket() -> Int32 {
        let f = socket(AF_UNIX, SOCK_STREAM, 0)
        guard f >= 0 else { return -1 }
        var on: Int32 = 1
        setsockopt(f, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        let cap = MemoryLayout.size(ofValue: addr.sun_path)
        guard bytes.count < cap else { close(f); return -1 }
        withUnsafeMutableBytes(of: &addr.sun_path) { buf in
            for (i, b) in bytes.enumerated() { buf[i] = b }
            buf[bytes.count] = 0
        }
        addr.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        let rc = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(f, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if rc != 0 { close(f); return -1 }
        return f
    }

    private func readLoop(_ f: Int32) {
        var pending = Data()
        var buf = [UInt8](repeating: 0, count: 16 * 1024)
        while isRunning {
            var pfd = pollfd(fd: f, events: Int16(POLLIN), revents: 0)
            let pr = poll(&pfd, 1, 1000)
            if pr < 0 { if errno == EINTR { continue }; return }
            if pr == 0 { continue }
            let n = Darwin.read(f, &buf, buf.count)
            if n <= 0 { return }
            pending.append(contentsOf: buf[0..<n])
            if pending.count > 1 << 20 { return }  // garbage guard
            while let nl = pending.firstIndex(of: 0x0A) {
                let line = pending.subdata(in: pending.startIndex..<nl)
                pending.removeSubrange(pending.startIndex...nl)
                guard !line.isEmpty,
                      let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
                DispatchQueue.main.async { self.onMessage?(obj) }
            }
        }
    }
}
