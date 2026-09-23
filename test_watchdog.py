#!/usr/bin/env python
"""
Test del WorkerWatchdog: verifica che monitori e riavvii i worker thread morti.

Testa:
  1. RabbitMQ connectivity (reale, CloudAMQP)
  2. _is_connection_open() con connessioni mock
  3. _reconnect_shared_connection() con connessione reale RabbitMQ
  4. Watchdog rileva worker morto e lo riavvia (real RabbitMQ + MockWorker)
  5. Watchdog riconnette shared_connection chiusa, poi riavvia (MockWorker)
  6. Clean shutdown via shutdown_event

Pre-requisiti:
  - Connessione internet (CloudAMQP)
  - pika installato (pip install pika)

Uso:
  cd vast
  python test_watchdog.py

  # Skippa il test di connettivita reale (solo test isolati):
  SKIP_RABBITMQ=1 python test_watchdog.py
"""

import os
import sys
import time
import threading

# Config
RABBITMQ_URL = os.environ.get(
    "RABBITMQ_URL",
    "amqps://broydvge:xPRe6nBrj4jfzMKtgbQxWJjZ5FjRVM0v@hawk.rmq.cloudamqp.com/broydvge"
)
SKIP_RABBITMQ = os.environ.get("SKIP_RABBITMQ", "0") == "1"
WATCHDOG_INTERVAL = 0.3  # secondi tra un check e l'altro

OK = "OK"
FAIL = "FAIL"
WARN = "WARN"

passed = 0
failed = 0


# ---- Helpers ----------------------------------------------------------------
def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    mark = "[PASS]" if condition else "[FAIL]"
    msg = f"  {mark} {label}"
    if detail:
        msg += f"  ->  {detail}"
    print(msg)
    if condition:
        passed += 1
    else:
        failed += 1
    return condition


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---- MockWorker (bloccante, usa stop_event) ----------------------------------
class MockWorker:
    """Mock di RabbitMqInferenceWorker con start() bloccante."""

    def __init__(self, shared_connection, skip_prewarm=False):
        self.shared_connection = shared_connection
        self.skip_prewarm = skip_prewarm
        self._stop_event = threading.Event()

    def start(self):
        """Simula il consumer loop: blocca fino a stop()."""
        self._stop_event.wait()

    def stop(self):
        """Sblocca start(), facendo uscire il thread."""
        self._stop_event.set()


# ---- 1. RabbitMQ connectivity -----------------------------------------------
def test_rabbitmq_connectivity():
    """Verifica che RabbitMQ broker sia raggiungibile."""
    section("RabbitMQ connectivity (real)")

    if SKIP_RABBITMQ:
        print(f"  [{WARN}] SKIP_RABBITMQ=1 - skipping")
        return True

    import pika
    try:
        params = pika.URLParameters(RABBITMQ_URL)
        conn = pika.BlockingConnection(params)
        print(f"  [PASS] RabbitMQ connected: is_open={conn.is_open}")
        conn.close()
        return True
    except Exception as e:
        print(f"  [FAIL] Cannot connect to RabbitMQ: {e}")
        return False


# ---- 2. Unit: _is_connection_open() -----------------------------------------
def test_is_connection_open():
    """Testa _is_connection_open con connessioni None, aperta, chiusa."""
    section("Unit: _is_connection_open()")

    from worker_watchdog import WorkerWatchdog

    shutdown = threading.Event()
    wd = WorkerWatchdog(workers=[], shared_connection=None, shutdown_event=shutdown)

    # Test 1: shared_connection = None
    check("_is_connection_open(None)", wd._is_connection_open() is False)

    # Test 2: connessione aperta
    class FakeOpen:
        is_open = True
    wd.shared_connection = FakeOpen()
    check("_is_connection_open(open=True)", wd._is_connection_open() is True)

    # Test 3: connessione chiusa
    class FakeClosed:
        is_open = False
    wd.shared_connection = FakeClosed()
    check("_is_connection_open(open=False)", wd._is_connection_open() is False)

    # Test 4: exception-safe
    class FakeBroken:
        @property
        def is_open(self):
            raise RuntimeError("broken")
    wd.shared_connection = FakeBroken()
    check("_is_connection_open(raises)", wd._is_connection_open() is False)


# ---- 3. Unit: _reconnect_shared_connection() (real RabbitMQ) -------------------
def test_reconnect_logic():
    """
    Testa _reconnect_shared_connection() con vera connessione RabbitMQ.
    Partiamo da connessione finta chiusa, il watchdog riconnette via RABBITMQ_URL.
    """
    section("Unit: _reconnect_shared_connection() (real RabbitMQ)")

    from worker_watchdog import WorkerWatchdog
    import pika

    if SKIP_RABBITMQ:
        print(f"  [{WARN}] SKIP_RABBITMQ=1 - skipping")
        return True

    shutdown = threading.Event()

    # Imposta RABBITMQ_URL nell'environment per _reconnect_shared_connection()
    old_env = os.environ.get("RABBITMQ_URL")
    os.environ["RABBITMQ_URL"] = RABBITMQ_URL

    class FakeClosedWithParams:
        is_open = False
        is_closed = True
        params = None  # forza fallback a RABBITMQ_URL env var

    wd = WorkerWatchdog(
        workers=[],
        shared_connection=FakeClosedWithParams(),
        shutdown_event=shutdown,
    )

    check("pre: _is_connection_open()", wd._is_connection_open() is False)

    try:
        result = wd._reconnect_shared_connection()
        check("_reconnect_shared_connection() succeeded", result is True)
        check("post: shared_connection.is_open", wd.shared_connection.is_open is True)
        # Cleanup
        if wd.shared_connection and not wd.shared_connection.is_closed:
            wd.shared_connection.close()
    except Exception as e:
        check("_reconnect_shared_connection()", False, str(e))
        return False
    finally:
        # Ripristina env var originale
        if old_env is not None:
            os.environ["RABBITMQ_URL"] = old_env
        else:
            os.environ.pop("RABBITMQ_URL", None)

    return True


# ---- 4. Integration: 2 worker, 1 muore, watchdog riavvia --------------------
def test_watchdog_restarts_dead_worker():
    """
    Scenario reale:
      - 2 MockWorker avviati su RabbitMQ reale
      - worker 0 muore
      - watchdog rileva e ricrea il thread
      - torniamo a 2 worker vivi
    """
    section("Integration: 2 workers, 1 dies, watchdog restarts")

    from worker_watchdog import WorkerWatchdog

    if SKIP_RABBITMQ:
        print(f"  [{WARN}] SKIP_RABBITMQ=1 - skipping")
        return True

    import pika

    # Connessione reale
    try:
        connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        print("  [INFO] Real RabbitMQ connection established")
    except Exception as e:
        check("Real RabbitMQ connection", False, str(e))
        return False

    try:
        shutdown = threading.Event()

        def factory(conn, skip_prewarm):
            return MockWorker(shared_connection=conn, skip_prewarm=skip_prewarm)

        # 2 worker vivi
        w_a = factory(connection, skip_prewarm=False)
        w_b = factory(connection, skip_prewarm=True)
        t_a = threading.Thread(target=w_a.start, daemon=True, name="worker-0")
        t_b = threading.Thread(target=w_b.start, daemon=True, name="worker-1")
        t_a.start()
        t_b.start()
        time.sleep(0.1)

        alive_before = [t_a.is_alive(), t_b.is_alive()]
        print(f"  [INFO] Workers alive before: {alive_before}")
        check("Both workers alive initially", all(alive_before))

        # Uccidi worker 0
        w_a.stop()
        t_a.join(timeout=2)
        check("Worker 0 killed", not t_a.is_alive())

        # Watchdog
        workers = [(t_a, w_a), (t_b, w_b)]
        wd = WorkerWatchdog(
            workers=workers,
            shared_connection=connection,
            shutdown_event=shutdown,
            worker_factory=factory,
        )
        wd._check_interval = WATCHDOG_INTERVAL

        wd_exceptions = []
        def run_wd():
            try:
                wd.run()
            except Exception as e:
                wd_exceptions.append(e)

        wd_thread = threading.Thread(target=run_wd, daemon=True, name="watchdog")
        wd_thread.start()

        wait = WATCHDOG_INTERVAL * 2 + 2.0
        print(f"  [INFO] Waiting {wait:.1f}s for watchdog to restart...")
        time.sleep(wait)

        alive_after = [t.is_alive() for (t, _w) in wd.workers]
        alive_count = sum(1 for a in alive_after if a)
        print(f"  [INFO] Workers alive after: {alive_after}")

        check("Worker count restored to 2",
              alive_count == 2,
              f"before=2, after={alive_count}")
        check("Watchdog no exceptions",
              len(wd_exceptions) == 0,
              f"exceptions={wd_exceptions}")

        # Shutdown
        shutdown.set()
        wd_thread.join(timeout=5)
        check("Watchdog stopped", not wd_thread.is_alive())

        w_b.stop()
        t_b.join(timeout=2)

    finally:
        try:
            if connection and not connection.is_closed:
                connection.close()
                print("  [INFO] Connection closed")
        except Exception:
            pass


# ---- 5. Integration: connessione chiusa -> reconnect -> restart --------------
def test_watchdog_reconnect_and_restart():
    """
    Scenario: shared_connection chiusa, worker morto.
    Watchdog riconnette e ricrea il worker.
    """
    section("Integration: closed conn -> reconnect -> restart")

    from worker_watchdog import WorkerWatchdog

    shutdown = threading.Event()

    # Imposta RABBITMQ_URL nell'environment per _reconnect_shared_connection()
    old_env = os.environ.get("RABBITMQ_URL")
    os.environ["RABBITMQ_URL"] = RABBITMQ_URL

    class FakeClosedConn:
        is_open = False
        is_closed = True
        params = None

        def close(self):
            pass

    fake_conn = FakeClosedConn()

    dead_worker = MockWorker(shared_connection=fake_conn, skip_prewarm=False)
    dead_thread = threading.Thread(target=lambda: None, daemon=True, name="dead-w")
    dead_thread.start()
    dead_thread.join()

    check("pre: dead thread", not dead_thread.is_alive())

    def factory(conn, skip_prewarm):
        return MockWorker(shared_connection=conn, skip_prewarm=skip_prewarm)

    workers = [(dead_thread, dead_worker)]
    wd = WorkerWatchdog(
        workers=workers,
        shared_connection=fake_conn,
        shutdown_event=shutdown,
        worker_factory=factory,
    )
    wd._check_interval = WATCHDOG_INTERVAL

    wd_exceptions = []
    def run_wd():
        try:
            wd.run()
        except Exception as e:
            wd_exceptions.append(e)

    wd_thread = threading.Thread(target=run_wd, daemon=True, name="watchdog")
    wd_thread.start()

    wait = WATCHDOG_INTERVAL * 2 + 5.0
    print(f"  [INFO] Waiting {wait:.1f}s for reconnect + restart...")
    time.sleep(wait)

    alive_after = [t.is_alive() for (t, _w) in wd.workers]
    alive_count = sum(1 for a in alive_after if a)
    print(f"  [INFO] Workers alive after: {alive_after}")

    is_open_now = wd._is_connection_open()
    check("Connection reconnected", is_open_now, f"is_open={is_open_now}")
    check("Dead worker replaced",
          alive_count >= 1,
          f"workers_alive={alive_count}/{len(wd.workers)}")

    shutdown.set()
    wd_thread.join(timeout=5)

    for _t, w in wd.workers:
        if isinstance(w, MockWorker):
            w.stop()

    try:
        if wd.shared_connection and not wd.shared_connection.is_closed:
            wd.shared_connection.close()
    except Exception:
        pass

    # Ripristina env var originale
    if old_env is not None:
        os.environ["RABBITMQ_URL"] = old_env
    else:
        os.environ.pop("RABBITMQ_URL", None)


# ---- 6. Unit: clean shutdown ------------------------------------------------
def test_clean_shutdown():
    """Verifica clean shutdown via shutdown_event."""
    section("Unit: clean shutdown")

    from worker_watchdog import WorkerWatchdog

    shutdown = threading.Event()

    class FakeConn:
        is_open = True
        is_closed = False
        params = None

    w_a = MockWorker(shared_connection=FakeConn(), skip_prewarm=False)
    w_b = MockWorker(shared_connection=FakeConn(), skip_prewarm=True)
    t_a = threading.Thread(target=w_a.start, daemon=True, name="w-a")
    t_b = threading.Thread(target=w_b.start, daemon=True, name="w-b")
    t_a.start()
    t_b.start()
    time.sleep(0.1)

    wd = WorkerWatchdog(
        workers=[(t_a, w_a), (t_b, w_b)],
        shared_connection=FakeConn(),
        shutdown_event=shutdown,
    )
    wd._check_interval = WATCHDOG_INTERVAL

    wd_ex = []
    def run_wd():
        try:
            wd.run()
        except Exception as e:
            wd_ex.append(e)

    wd_thread = threading.Thread(target=run_wd, daemon=True, name="watchdog")
    wd_thread.start()
    time.sleep(0.2)

    shutdown.set()
    wd_thread.join(timeout=5)

    check("Watchdog stopped", not wd_thread.is_alive())
    check("No exceptions", len(wd_ex) == 0, f"ex={wd_ex}")

    w_a.stop()
    w_b.stop()


# ---- Main -------------------------------------------------------------------
def main():
    global passed, failed

    print("\n" + "=" * 60)
    print("  WorkerWatchdog - Test Suite")
    print(f"  RABBITMQ_URL: {RABBITMQ_URL.split('@')[0]}@...")
    print(f"  SKIP_RABBITMQ: {SKIP_RABBITMQ}")
    print("=" * 60)

    test_rabbitmq_connectivity()

    # Unit (sempre)
    test_is_connection_open()
    test_reconnect_logic()
    test_clean_shutdown()

    # Integration (richiede RabbitMQ)
    test_watchdog_restarts_dead_worker()
    test_watchdog_reconnect_and_restart()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()