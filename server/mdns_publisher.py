import asyncio
import socket
from settings import MDNS_SERVICE_NAME, MDNS_SERVICE_TYPE, MDNS_TXT_RECORD

try:
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:
    # pip install zeroconf
    ServiceInfo = None
    Zeroconf = None


class MDNSPublisher:
    SERVICE_TYPE = MDNS_SERVICE_TYPE
    SERVICE_NAME = MDNS_SERVICE_NAME

    def __init__(self, port: int):
        self.port = port
        self.ip = self._detect_local_ip()
        self.zeroconf = None
        self.service_info = None

    @staticmethod
    def _detect_local_ip() -> str:
        print("Detecting local IP address for mDNS...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # UDP connect() here does not send traffic; it only selects a local interface.
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            try:
                return socket.gethostbyname(socket.gethostname())
            except OSError:
                return "127.0.0.1"
        finally:
            sock.close()

    def _register(self):
        if Zeroconf is None or ServiceInfo is None:
            print("mDNS disabled: zeroconf package is missing.")
            return

        properties = {
            key.encode("utf-8"): value.encode("utf-8")
            for key, value in MDNS_TXT_RECORD.items()
        }
        full_name = f"{self.SERVICE_NAME}.{self.SERVICE_TYPE}"
        self.zeroconf = Zeroconf()
        self.service_info = ServiceInfo(
            type_=self.SERVICE_TYPE,
            name=full_name,
            addresses=[socket.inet_aton(self.ip)],
            port=self.port,
            properties=properties,
        )
        self.zeroconf.register_service(self.service_info)
        print(f"mDNS registered: {self.ip}:{self.port}")

    async def start(self):
        await asyncio.to_thread(self._register)

    def _unregister(self):
        if self.zeroconf is None or self.service_info is None:
            return
        try:
            self.zeroconf.unregister_service(self.service_info)
        finally:
            self.zeroconf.close()
            self.zeroconf = None
            self.service_info = None

    async def stop(self):
        await asyncio.to_thread(self._unregister)
