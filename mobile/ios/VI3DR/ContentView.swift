//
//  ContentView.swift
//  VI3DR
//
//  Created by Piotr Marcol on 09/01/2026.
//

import SwiftUI
import AVFoundation
import VideoToolbox

struct ContentView: View {
    @StateObject private var streamer = CameraStreamer()
    
    @State private var host: String = "192.168.1.110"
    @State private var port: String = "8765"
    @State private var jpegQuality: Double = 0.6
    @State private var targetFPS: Double = 15
    
    
    var body: some View {
        VStack(spacing: 12) {
            ZStack(alignment: .topLeading){
                CameraPreview(session: streamer.session)
                    .onAppear {streamer.start()}
                    .onDisappear {streamer.stop()}
                
                VStack(alignment: .leading, spacing: 6) {
                    Text(streamer.isConnected ? "WS: Connected" : "WS: disconnected")
                        .padding(6)
                        .background(.black.opacity(0.6))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    
                    Text("Sent FPS: \(String(format: "%.1f", streamer.sentFPS))")
                        .padding(6)
                        .background(.black.opacity(0.6))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .padding(10)
            }
            .frame(maxHeight: 420)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            
            HStack {
                TextField("Host (IP laptopa)", text: $host)
                                    .textFieldStyle(.roundedBorder)
                                    .keyboardType(.numbersAndPunctuation)

                                TextField("Port", text: $port)
                                    .textFieldStyle(.roundedBorder)
                                    .keyboardType(.numberPad)
                                    .frame(width: 90)
            }
            
            VStack(alignment: .leading) {
                HStack {
                    Text("JPEG quality")
                    Spacer()
                    Text(String(format: "%.2f", jpegQuality))
                        .monospacedDigit()
                }
                Slider(value: $jpegQuality, in: 0.2...0.9, step: 0.05)
            }
            
            VStack(alignment: .leading) {
                HStack {
                    Text("Target FPS")
                    Spacer()
                    Text("\(Int(targetFPS))")
                        .monospacedDigit()
                }
                Slider(value: $targetFPS, in: 5...60, step: 1)
            }
            
            HStack(spacing: 12) {
                Button(streamer.isConnected ? "Disconnect" : "Connect"){
                    if streamer.isConnected {
                        streamer.disconnect()
                    } else {
                        streamer.jpegQuality = Float(jpegQuality)
                        streamer.targetFPS = targetFPS
                        streamer.connect(host: host, port: Int(port) ?? 8765)
                    }
                }
                .buttonStyle(.borderedProminent)
                
                Button("Restart Camera") {
                    streamer.stop()
                    streamer.start()
                }
                .buttonStyle(.bordered)
            }
            Spacer()
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
