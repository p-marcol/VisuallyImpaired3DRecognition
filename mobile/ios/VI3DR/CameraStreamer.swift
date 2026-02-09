//
//  CameraStreamer.swift
//  VI3DR
//
//  Created by Piotr Marcol on 09/01/2026.
//

import SwiftUI
import AVFoundation
import VideoToolbox
internal import Combine

final class CameraStreamer: NSObject, ObservableObject {
    let session = AVCaptureSession()
    
    @Published var isConnected: Bool = false
    @Published var sentFPS: Double = 0
    
    var jpegQuality: Float = 0.6
    var targetFPS: Double = 15
    
    private let videoOutput = AVCaptureVideoDataOutput()
        private let captureQueue = DispatchQueue(label: "capture.queue", qos: .userInitiated)

        private var wsTask: URLSessionWebSocketTask?
        private var urlSession: URLSession = URLSession(configuration: .default)

        // FPS throttling + stats
        private var lastSentTime: CFTimeInterval = 0
        private var sentFramesInWindow: Int = 0
        private var windowStart: CFTimeInterval = CACurrentMediaTime()

        // CIContext for fast-ish conversion
        private let ciContext = CIContext()

    func start() {
        requestCameraPermissionsIfNeeded { [weak self] granted in
            guard let self, granted else {return}
            self.configureSessionIfNeeded()
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }
    
    func stop() {
        if session.isRunning {session.stopRunning()}
    }
    
    func connect(host: String, port: Int) {
        disconnect()
        
        // ws://IP:PORT
        guard let url = URL(string: "ws://\(host):\(port)") else {return}
        let task = urlSession.webSocketTask(with: url)
        wsTask = task
        task.resume()
        
        DispatchQueue.main.async {
            self.isConnected = true
        }
        receiveLoop()
    }
    
    func disconnect() {
        wsTask?.cancel(with: .goingAway, reason: nil)
        wsTask = nil
        DispatchQueue.main.async {self.isConnected = false}
    }
    
    private func receiveLoop() {
        guard let wsTask else {return}
        wsTask.receive { [weak self] result in
            guard let self else {return}
            switch result {
            case .success:
                self.receiveLoop()
            case .failure:
                self.disconnect()
            }
        }
    }
    
    private func configureSessionIfNeeded() {
        guard session.inputs.isEmpty else {return}
        
        session.beginConfiguration()
        
        // preset: 720p
        if session.canSetSessionPreset(.hd1280x720) {
            session.sessionPreset = .hd1280x720
        } else {
            session.sessionPreset = .high
        }
        
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: captureQueue)
        
        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(videoOutput)
        
        if let conn = videoOutput.connection(with: .video) {
            if conn.isVideoOrientationSupported {
                conn.videoOrientation = .portrait
            }
        }
        
        session.commitConfiguration()
    }
    
    private func requestCameraPermissionsIfNeeded(_ completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        default:
            completion(false)
        }
    }
    
    private func shouldSendFrame(now: CFTimeInterval) -> Bool {
        let minInterval = 1.0 / max(targetFPS, 1.0)
        if now - lastSentTime >= minInterval {
            lastSentTime = now
            return true
        }
        return false
    }
    
    private func updateSendFPS(now: CFTimeInterval) {
        if now - windowStart >= 1.0 {
            sentFPS = Double(sentFramesInWindow) / (now - windowStart)
            windowStart = now
            sentFramesInWindow = 0
        }
    }
    
    private func sendJPEG(_ data: Data) {
        guard let wsTask else {return}
        wsTask.send(.data(data)) {[weak self] error in
            if error != nil {
                self?.disconnect()
            }
        }
    }
    
    private func jpegData(from pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: CGFloat(jpegQuality))
    }
}

extension CameraStreamer: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard isConnected else {return}
        let now = CACurrentMediaTime()
        guard shouldSendFrame(now: now) else {return}
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return}
        guard let jpeg = jpegData(from: pixelBuffer) else {return}
        
        sentFramesInWindow += 1
        updateSendFPS(now: now)
        sendJPEG(jpeg)
    }
}
