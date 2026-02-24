# WI3DR KMP Mobile Client (Android + iOS)

Mobilna część systemu `VisuallyImpaired3DRecognition` zbudowana w Kotlin Multiplatform (KMP) + Compose Multiplatform.

Aplikacja:
- przechwytuje obraz z kamery (natywnie na Android/iOS),
- koduje klatki do JPEG,
- wysyła je przez WebSocket (`ws://`) do serwera,
- pozwala ustawić `IP`, `port` i docelowy limit `FPS`,
- pokazuje status połączenia i stan `LIVE`.

## Zakres tej części systemu

To repo (`KMP/`) dokumentuje tylko klient mobilny:
- `Android` (CameraX + Compose)
- `iOS` (AVFoundation + SwiftUI host + Compose)

Serwer rozpoznawania działa poza tym repo i nie jest tutaj opisywany.

## Najważniejsze funkcje

- Wspólna logika streamingu w `commonMain`:
  - WebSocket client
  - stan UI (`IP`, `port`, `FPS`, status połączenia)
  - throttling FPS
  - obsługa błędów połączenia
- Natywna kamera per platforma:
  - Android: CameraX
  - iOS: AVFoundation
- Ten sam UX na Android i iOS:
  - panel sterowania + podgląd
  - status połączenia
  - badge `LIVE` w prawym górnym rogu preview
- Wymuszony tryb landscape
- Proporcje preview/kamery 4:3 (wymóg badawczy)

## Architektura (skrót)

### Wspólna warstwa (`commonMain`)

- `StreamingController`
  - zarządza stanem streamingu, połączeniem WebSocket i limitem FPS
  - przyjmuje klatki z warstwy platformowej
- `StreamingUiState`
  - stan formularza i status połączenia
- `FrameSocketClient`
  - wysyłka binarnych ramek JPEG przez WebSocket
- `App`
  - wspólny UI panelu sterowania

### Android (`androidMain`)

- `MainActivity`:
  - layout (panel + preview)
  - CameraX setup
  - toast na błąd połączenia
- `CameraAnalyzer`:
  - pobranie klatki i przekazanie do `StreamingController`
- `ImageProxy.toJpegBytes()`:
  - konwersja klatki do JPEG

### iOS (`iosMain` + `iosApp`)

- `IOSApp` (Compose):
  - layout jak Android (panel + preview)
  - `LIVE` badge
  - banner błędu połączenia (toast-like)
- `IosCameraManager` (AVFoundation):
  - `AVCaptureSession`
  - preview (`AVCaptureVideoPreviewLayer`)
  - capture callback
  - asynchroniczne kodowanie JPEG (osobna kolejka encode)
- `iosApp` (SwiftUI/Xcode):
  - host aplikacji iOS osadzający `ComposeUIViewController`

## Struktura projektu

- `composeApp/`
  - moduł KMP (common + androidMain + iosMain)
- `iosApp/`
  - projekt Xcode (host iOS)
- `gradle/`, `gradlew`
  - build tooling

## Wymagania

### Ogólne

- JDK 11+
- Gradle Wrapper (w repo)

### Android

- Android Studio (aktualne)
- Android SDK zgodne z konfiguracją projektu
- urządzenie/emulator z kamerą

### iOS

- macOS
- Xcode
- iPhone do realnych testów kamery (simulator ma ograniczenia)
- konto Apple Developer (nawet darmowe do testu lokalnego)

## Szybki start

### Android (CLI)

```bash
./gradlew :composeApp:assembleDebug
```

Opcjonalnie szybka weryfikacja kompilacji:

```bash
./gradlew :composeApp:compileDebugKotlinAndroid
```

### iOS (CLI)

Kompilacja modułu KMP dla iOS simulator:

```bash
./gradlew :composeApp:compileKotlinIosSimulatorArm64
```

Uruchomienie aplikacji iOS wykonuje się z Xcode (patrz sekcja niżej).

## Uruchomienie iOS na iPhonie (Xcode)

1. Otwórz `iosApp/iosApp.xcodeproj` w Xcode.
2. Wybierz target `iosApp`.
3. W `Signing & Capabilities`:
   - włącz `Automatically manage signing`
   - wybierz swój `Team`
   - ustaw unikalny `Bundle Identifier` (jeśli Xcode zgłasza konflikt)
4. Wybierz podłączonego iPhone’a jako device run target.
5. Kliknij `Run` (`Cmd+R`).

Przy pierwszym uruchomieniu:
- zaakceptuj dostęp do kamery,
- zaakceptuj dostęp do sieci lokalnej.

## Konfiguracja połączenia (w aplikacji)

W panelu sterowania ustaw:
- `IP Address` serwera (np. `192.168.x.x`)
- `Port` serwera (np. `8765`)
- `FPS` (docelowy limit wysyłki)

Następnie kliknij `Start Streaming`.

### Ważne

- Telefon i serwer muszą być w tej samej sieci lokalnej.
- Aplikacja używa `ws://` (cleartext WebSocket), nie `wss://`.
- iOS ma odpowiednio skonfigurowany `Info.plist` pod lokalną sieć i ATS.

## UI / obsługa

- `Status` pokazuje stan połączenia:
  - `Disconnected`
  - `Connecting...`
  - `Connected`
  - `Connection failed`
- `LIVE` pojawia się tylko po faktycznym nawiązaniu połączenia (`Connected`)
- Błędy połączenia:
  - Android: `Toast`
  - iOS: banner nad preview (toast-like)

### Pola IP / Port

- `Enter` nie dodaje nowej linii (`singleLine`)
- `Enter/Done` chowa klawiaturę i zdejmuje focus
- Android:
  - IP: klawiatura `Decimal`
  - Port: klawiatura `Number`
- iOS:
  - używana jest klawiatura z `Enter` (ASCII), bo iOS `numberPad/decimalPad` zwykle nie pokazuje `Done`
  - znaki są filtrowane programowo:
    - IP: tylko cyfry i `.`
    - Port: tylko cyfry

## Kamera i orientacja

### Android

- `MainActivity` wymuszona w `landscape` (AndroidManifest)
- CameraX preview + analiza klatek

### iOS

- Aplikacja wymuszona do `landscape` (`Info.plist`)
- `UIRequiresFullScreen = true`
- Kamera i preview AVFoundation wymuszone na `landscape`
- Preview w UI ma proporcję `4:3`
- Preset kamery ustawiony na `640x480` (4:3)

## Wydajność i ograniczenia (ważne do dokumentacji badań)

### FPS suwaka = limit docelowy, nie gwarancja

Suwak `FPS` ustawia górny limit wysyłki, ale rzeczywisty FPS zależy od:
- wydajności kamery
- kosztu kodowania JPEG
- CPU urządzenia
- sieci
- wydajności serwera

### iOS może mieć niższy FPS niż Android

Mimo zachowania tej samej logiki streamingu:
- pipeline kamery/enkodowania na iOS i Androidzie różni się technologicznie,
- iOS w tej implementacji może osiągać niższy FPS przy zachowaniu stabilności i poprawności działania.

To jest znane ograniczenie implementacyjne i warto je opisać w dokumentacji eksperymentu/badań.

### Priorytet stabilności preview na iOS

Aby nie blokować podglądu:
- kodowanie JPEG na iOS zostało przeniesione poza callback kamery (osobna kolejka encode),
- stosowany jest limit `1 in-flight` dla encodera (zrzucanie nadmiarowych klatek pod obciążeniem).

To poprawia płynność podglądu kosztem maksymalnego FPS wysyłki.

## Troubleshooting

### iOS: brak obrazu / czarny preview

- sprawdź uprawnienie do kamery
- testuj na fizycznym urządzeniu (nie simulator)
- sprawdź czy aplikacja ma dostęp do kamery w `Settings > Privacy`

### iOS: brak połączenia z serwerem

- sprawdź czy iPhone i serwer są w tej samej sieci
- sprawdź IP/port
- sprawdź czy serwer nasłuchuje na `ws://`
- sprawdź zgodę na `Local Network` na iPhonie

### iOS: obraz jest poziomy, ale odwrócony (lewo/prawo)

Zmień orientację w `IosCameraManager`:
- `AVCaptureVideoOrientationLandscapeRight`
- na `AVCaptureVideoOrientationLandscapeLeft`

### Android/iOS: niskie FPS

- to może być normalne ograniczenie pipeline’u
- suwak `FPS` nie gwarantuje osiągnięcia wartości
- najpierw sprawdź, czy obraz jest wysyłany stabilnie i czy `LIVE` jest aktywne

## Znane ostrzeżenia builda

- `expect/actual` classes warning (beta) dla `PlatformKeyboardHints`
  - build przechodzi poprawnie
- ostrzeżenie KMP + `com.android.application` w jednym module (`composeApp`)
  - obecnie nie blokuje działania
  - docelowo warto rozdzielić moduł KMP i moduł aplikacji Android

## Przydatne komendy (weryfikacja)

```bash
# Android Kotlin compile
./gradlew :composeApp:compileDebugKotlinAndroid

# iOS Kotlin compile (simulator target)
./gradlew :composeApp:compileKotlinIosSimulatorArm64
```

## Dalszy rozwój (opcjonalnie)

- jawny status uprawnień kamery na iOS (np. denied / unavailable)
- licznik `actual FPS` (preview vs send FPS)
- metryki wydajności dla eksperymentów (CPU, encode time, send time)
- wsparcie `wss://` (TLS)

