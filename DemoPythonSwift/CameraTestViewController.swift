//
//  CameraViewController.swift
//  DemoPythonSwift
//
//  Created by Lai Minh on 30/10/25.
//

import UIKit
import AVFoundation
import TensorFlowLite



class CameraTestViewController: UIViewController {
    
    // MARK: - Drawing layers
    private var boundingBoxLayers = [CAShapeLayer]()
    private let labelLayer = CATextLayer()
    // MARK: - UI
    private let previewView = UIView()
    private let predictionLabel: UILabel = {
        let l = UILabel()
        l.translatesAutoresizingMaskIntoConstraints = false
        l.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        l.textColor = .white
        l.font = UIFont.systemFont(ofSize: 20, weight: .bold)
        l.numberOfLines = 0
        l.textAlignment = .left
        l.layer.cornerRadius = 8
        l.clipsToBounds = true
        l.text = "Predictions will appear here"
        return l
    }()
    
    // **[NEW]** FPS Display Label
    private let fpsLabel: UILabel = {
        let l = UILabel()
        l.translatesAutoresizingMaskIntoConstraints = false
        l.backgroundColor = UIColor.red.withAlphaComponent(0.6)
        l.textColor = .white
        l.font = UIFont.systemFont(ofSize: 18, weight: .bold)
        l.textAlignment = .center
        l.layer.cornerRadius = 4
        l.clipsToBounds = true
        l.text = "FPS: -"
        return l
    }()
    
    // MARK: - Camera
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer!
    
    // MARK: - TFLite
    private var interpreter: Interpreter!
    private let inputWidth = 224
    private let inputHeight = 224
    private let inputChannels = 3
    
    
    private var labels: [String] = []
    
    // throttle frames
    private var lastRun: Date = .distantPast
    private let minFrameInterval: TimeInterval = 0.05
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupUI()
        setupCamera()
        loadModel()
        loadLabels()
    }
    
    private func setupUI() {
        previewView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(previewView)
        
        // **[NEW]** Add FPS Label
        view.addSubview(fpsLabel)
        view.addSubview(predictionLabel)
        
        
        NSLayoutConstraint.activate([
            previewView.topAnchor.constraint(equalTo: view.topAnchor),
            previewView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            previewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            previewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            // **[NEW]** FPS Label Constraints (Top Left)
            fpsLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            fpsLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 10),
            fpsLabel.widthAnchor.constraint(equalToConstant: 120),
            fpsLabel.heightAnchor.constraint(equalToConstant: 30),
            
            // **[NEW]** Prediction Label Constraints (Top Right)
            predictionLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            predictionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -10),
            predictionLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 40)
        ])
    }
    
    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high
        
        // Select the back camera
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else{
            print("No back camera.")
            return
        }
        
        guard let input = try? AVCaptureDeviceInput(device: camera) else {
            print("Can't create input from camera")
            return
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        // Video output
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_32BGRA
        ]
        dataOutput.alwaysDiscardsLateVideoFrames = true
        
        let queue = DispatchQueue(label: "videoQueue")
        dataOutput.setSampleBufferDelegate(self, queue: queue)
        
        if captureSession.canAddOutput(dataOutput) {
            captureSession.addOutput(dataOutput)
        }
        
        // Orient preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = previewView.bounds
        previewView.layer.addSublayer(previewLayer)
        
        // Connection orientation
        if let connection = dataOutput.connection(with: .video),
           connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
        
        DispatchQueue.global(qos: .background).async {
            self.captureSession.commitConfiguration()
            self.captureSession.startRunning()
        }
    }
    
    private func loadLabels() {
        guard let labelsPath = Bundle.main.path(forResource: "labels", ofType: "txt"),
              let content = try? String(contentsOfFile: labelsPath) else {
            print("Labels not found. Predictions will show indices.")
            return
        }
        labels = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        print("Loaded \(labels.count) labels.")
    }
    
    private func loadModel() {
        // Make sure model file exists in bundle
        guard let modelPath = Bundle.main.path(forResource: "efficientnet_b0_aug", ofType: "tflite") else {
            fatalError("Model file not found in bundle.")
        }
        
        do {
            // Create interpreter with options if needed (threads)
            var options = Interpreter.Options()
            options.threadCount = 2
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // 2. Allocate memory
            try interpreter.allocateTensors()
            print("TFLite interpreter created and tensors allocated.")
        } catch {
            fatalError("Failed to create interpreter: \(error)")
        }
    }
    
    private func runModel(on pixelBuffer: CVPixelBuffer) {
        // Throttle FPS to reduce CPU
        let now = Date()
        guard now.timeIntervalSince(lastRun) >= minFrameInterval else { return }
        lastRun = now
        
        // Get start time for inference calculation
        let inferenceStartTime = CFAbsoluteTimeGetCurrent()
        
        // Chuyển ảnh thành định dạng đầu vào model cần (crop 1/3 trung tâm, resize 224x224)
        guard let inputData = preprocess(pixelBuffer: pixelBuffer) else { return }
        
        do {
            // 4. Copy input
            try interpreter?.copy(inputData, toInputAt: 0)
            
            //   // 5. Run model
            try interpreter?.invoke()
            
            let inferenceEndTime = CFAbsoluteTimeGetCurrent()
            let inferenceTime = inferenceEndTime - inferenceStartTime
            let fps = 1.0 / inferenceTime
            
            // Output is handled in the background thread below
            
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self, let interpreter = self.interpreter else { return }
                
                do {
                    let outputTensor = try interpreter.output(at: 0)
                    
                    // 6. Interpret output as float array
                    let results = [Float](unsafeData: outputTensor.data) ?? []
                    
                    
                    let topResult = results.topK(k: 1).first
                    
                    print("Inference Time: \(inferenceTime * 1000) ms, FPS: \(fps), Result: \(String(describing: topResult))")
                    
                    // Prepare output text
                    var outputText = ""
                    if let result = topResult {
                        let label: String
                        if result.index < self.labels.count {
                            label = self.labels[result.index]
                        } else {
                            label = "Index \(result.index)"
                        }
                        
                        
                        let confidenceText = String(format: "%.1f", result.score * 100.0)
                        outputText = "\(label) (\(confidenceText)%)"
                    } else {
                        outputText = "No result"
                    }
                    
                    // 7. Update UI on the main thread
                    DispatchQueue.main.async {
                        self.predictionLabel.text = outputText
                        
                        // **[NEW]** Update FPS Label
                        self.fpsLabel.text = String(format: "FPS: %.1f", fps)
                    }
                } catch {
                    print("Lỗi khi xử lý output: \(error)")
                }
            }
        } catch {
            print("Lỗi khi chạy model: \(error)")
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = previewView.bounds
    }
    
    // MARK: - Bounding Box Drawing
    private func drawBoundingBox(cropRect: CGRect, originalSize: CGSize) {
        // 1. Remove previous box
        boundingBoxLayers.forEach { $0.removeFromSuperlayer() }
        boundingBoxLayers.removeAll()
        
        // 2. Map the cropRect (based on CVPixelBuffer size) to the previewLayer (screen size)
        // AVMakeRect ensures the mapping respects the aspect ratio of the video layer (resizeAspectFill)
        let videoRect = AVMakeRect(aspectRatio: originalSize, insideRect: previewView.bounds)
        
        // Calculate the scale and offset
        let scaleX = videoRect.width / originalSize.width
        let scaleY = videoRect.height / originalSize.height
        
        // Adjusted rect on screen
        let x = videoRect.origin.x + cropRect.origin.x * scaleX
        let y = videoRect.origin.y + cropRect.origin.y * scaleY
        let w = cropRect.width * scaleX
        let h = cropRect.height * scaleY
        
        let scaledRect = CGRect(x: x, y: y, width: w, height: h)
        
        // 3. Draw the box
        let boxLayer = CAShapeLayer()
        boxLayer.frame = scaledRect
        boxLayer.borderColor = UIColor.yellow.cgColor // Yellow (0, 255, 255 in BGR)
        boxLayer.borderWidth = 3.0
        boxLayer.fillColor = UIColor.clear.cgColor
        
        previewView.layer.addSublayer(boxLayer)
        boundingBoxLayers.append(boxLayer)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraTestViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        runModel(on: pixelBuffer) // Run model on each frame
        
    }
    
    
    // MARK: - Preprocess image (No normalization, only crop + resize)
    private func preprocess(pixelBuffer: CVPixelBuffer) -> Data? {
        let originalWidth = CVPixelBufferGetWidth(pixelBuffer)
        let originalHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        // 1. Calculate center 1/3 crop (box_w, box_h = w//3, h//3)
        let boxW = originalWidth / 3
        let boxH = originalHeight / 3
        let x1 = (originalWidth - boxW) / 2
        let y1 = (originalHeight - boxH) / 2
        let cropRect = CGRect(x: x1, y: y1, width: boxW, height: boxH)
        
        // 2. Crop CIImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).cropped(to: cropRect)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            print("ERROR: Failed to create CGImage from cropped CIImage.")
            return nil
        }
        
        // 3. Convert to UIImage and resize to model input size
        let croppedUIImage = UIImage(cgImage: cgImage)
        UIGraphicsBeginImageContext(CGSize(width: inputWidth, height: inputHeight))
        croppedUIImage.draw(in: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        guard let finalImage = resizedImage,
              let cgFinal = finalImage.cgImage else {
            print("ERROR: Resize failed.")
            return nil
        }
        
        // 4. Convert to raw RGB data (no normalization)
        let width = cgFinal.width
        let height = cgFinal.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        guard let colorSpace = cgFinal.colorSpace else { return nil }
        guard let context2 = CGContext(data: &pixelData,
                                       width: width,
                                       height: height,
                                       bitsPerComponent: bitsPerComponent,
                                       bytesPerRow: bytesPerRow,
                                       space: colorSpace,
                                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            print("ERROR: CGContext creation failed.")
            return nil
        }
        
        context2.draw(cgFinal, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // 5. Convert RGBA -> RGB and cast to Float32 (no normalization)
        var rgbArray = [Float32]()
        rgbArray.reserveCapacity(width * height * 3)
        for i in stride(from: 0, to: pixelData.count, by: 4) {
            let r = Float32(pixelData[i])
            let g = Float32(pixelData[i + 1])
            let b = Float32(pixelData[i + 2])
            rgbArray.append(r)
            rgbArray.append(g)
            rgbArray.append(b)
        }
        
        // 6. Draw bounding box on the screen (Main thread required for UI)
        DispatchQueue.main.async {
            self.drawBoundingBox(cropRect: cropRect,
                                 originalSize: CGSize(width: originalWidth, height: originalHeight))
        }
        
        print("preprocess(): returning raw RGB data, count=\(rgbArray.count)")
        return Data(buffer: UnsafeBufferPointer(start: rgbArray, count: rgbArray.count))
    }
}

