//
//  CameraPreview.swift
//  VI3DR
//
//  Created by Piotr Marcol on 09/01/2026.
//

import SwiftUI
import AVFoundation
import VideoToolbox

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = view.bounds
        view.layer.addSublayer(layer)
        
        context.coordinator.layer = layer
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.layer?.frame = uiView.bounds
    }
    
    func makeCoordinator() -> Coordinator { Coordinator() }
    
    
    class Coordinator {
        var layer: AVCaptureVideoPreviewLayer?
    }
}
