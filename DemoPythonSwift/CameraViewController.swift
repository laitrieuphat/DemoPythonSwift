////
////  CameraViewController.swift
////  DemoPythonSwift
////
////  Created by Lai Minh on 30/10/25.
////
//
//import UIKit
//import AVFoundation
//import TensorFlowLite
//
//class CameraViewController: UIViewController {
//    
//    // MARK: - Drawing layers
//    private var boundingBoxLayers = [CAShapeLayer]()
//    private let labelLayer = CATextLayer()
//    // MARK: - UI
//    private let previewView = UIView()
//    private let predictionLabel: UILabel = {
//        let l = UILabel()
//        l.translatesAutoresizingMaskIntoConstraints = false
//        l.backgroundColor = UIColor.black.withAlphaComponent(0.5)
//        l.textColor = .white
//        l.font = UIFont.systemFont(ofSize: 16, weight: .semibold)
//        l.numberOfLines = 2
//        l.textAlignment = .center
//        l.layer.cornerRadius = 8
//        l.clipsToBounds = true
//        return l
//    }()
//    
//    // MARK: - Camera
//    private let captureSession = AVCaptureSession()
//    private var previewLayer: AVCaptureVideoPreviewLayer!
//    
//    // MARK: - TFLite
//    private var interpreter: Interpreter!
//    private let inputWidth = 224
//    private let inputHeight = 224
//    private let inputChannels = 3
//    private var labels: [String] = []
//    
//    // throttle frames
//    private var lastRun: Date = .distantPast
//    private let minFrameInterval: TimeInterval = 0.20 // 5 FPS
//    
//    override func viewDidLoad() {
//        super.viewDidLoad()
//        view.backgroundColor = .black
//        setupUI()
//        setupCamera()
//        loadModel()
//        loadLabels()
//    }
//    
//    private func setupUI() {
//        previewView.translatesAutoresizingMaskIntoConstraints = false
//        view.addSubview(previewView)
//        view.addSubview(predictionLabel)
//        
//        NSLayoutConstraint.activate([
//            previewView.topAnchor.constraint(equalTo: view.topAnchor),
//            previewView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
//            previewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
//            previewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
//            
//            predictionLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
//            predictionLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
//            predictionLabel.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.9),
//            predictionLabel.heightAnchor.constraint(equalToConstant: 60)
//        ])
//    }
//    
//    private func setupCamera() {
//        captureSession.beginConfiguration()
//        captureSession.sessionPreset = .high
//        
//        // Select the back camera
//        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera,
//                                                   for: .video,
//                                                   position: .back) else {
//            print("No back camera.")
//            return
//        }
//        
//        guard let input = try? AVCaptureDeviceInput(device: camera) else {
//            print("Can't create input from camera")
//            return
//        }
//        
//        if captureSession.canAddInput(input) {
//            captureSession.addInput(input)
//        }
//        
//        // Video output
//        let dataOutput = AVCaptureVideoDataOutput()
//        dataOutput.videoSettings = [
//            kCVPixelBufferPixelFormatTypeKey as String:
//                kCVPixelFormatType_32BGRA
//        ]
//        dataOutput.alwaysDiscardsLateVideoFrames = true
//        
//        let queue = DispatchQueue(label: "videoQueue")
//        dataOutput.setSampleBufferDelegate(self, queue: queue)
//        
//        if captureSession.canAddOutput(dataOutput) {
//            captureSession.addOutput(dataOutput)
//        }
//        
//        // Orient preview layer
//        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
//        previewLayer.videoGravity = .resizeAspectFill
//        previewLayer.frame = previewView.bounds
//        previewView.layer.addSublayer(previewLayer)
//        
//        // Connection orientation
//        if let connection = dataOutput.connection(with: .video),
//           connection.isVideoOrientationSupported {
//            connection.videoOrientation = .portrait
//        }
//        
//        DispatchQueue.global(qos: .background).async {
//            self.captureSession.commitConfiguration()
//            self.captureSession.startRunning()
//        }
//        
//    }
//    
//    private func loadModel() {
//        // Make sure model file exists in bundle
//        guard let modelPath = Bundle.main.path(forResource: "efficientnet_b0_aug", ofType: "tflite") else {
//            fatalError("Model file not found in bundle.")
//        }
//        
//        do {
//            // Create interpreter with options if needed (threads)
//            var options = Interpreter.Options()
//                options.threadCount = 2
//            interpreter = try Interpreter(modelPath: modelPath, options: options)
//            // 2. Allocate memory
//            try interpreter.allocateTensors()
//            print("TFLite interpreter created and tensors allocated.")
//        } catch {
//            fatalError("Failed to create interpreter: \(error)")
//        }
//    }
//    
//    private func loadLabels() {
//        guard let labelsPath = Bundle.main.path(forResource: "labels", ofType: "txt"),
//              let content = try? String(contentsOfFile: labelsPath) else {
//            print("Labels not found. Predictions will show indices.")
//            return
//        }
//        labels = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
//        print("Loaded \(labels.count) labels.")
//    }
//    
//    private func runModel(on pixelBuffer: CVPixelBuffer) {
//        // Throttle FPS to reduce CPU
//        let now = Date()
//        guard now.timeIntervalSince(lastRun) >= minFrameInterval else { return }
//        lastRun = now
//        
//        // Convert pixelBuffer to UIImage -> resize -> Data
//        
//        // Chuyển ảnh thành định dạng đầu vào model cần
//        // (Ví dụ model cần float32 224x224)
//        guard let inputData = preprocess(pixelBuffer: pixelBuffer) else { return }
//        
//        do {
//            
//            // 4. Copy input
//            try interpreter?.copy(inputData, toInputAt: 0)
//            
//            //   // 5. Run model
//            try interpreter?.invoke()
//            
//            //
//            let outputTensor = try interpreter?.output(at: 0)
//            
//            // Giải thích kết quả
//            let results = [Float](unsafeData: outputTensor!.data) ?? []
//            DispatchQueue.main.async {
//                print("Kết quả phân tích: \(results)")
//            }
//        } catch {
//            print("Lỗi khi chạy model: \(error)")
//        }
//        
//        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
//            guard let self = self else { return }
//            do {
//                let outputTensor = try interpreter?.output(at: 0)
//                
//                // Interpret output as float array
//                let results = [Float](unsafeData: outputTensor!.data) ?? []
//                // Find top1
//                
//                DispatchQueue.main.async {
//                    print("Kết quả phân tích: \(results)")
//                    if let (idx, score) = results.argmaxWithScore() {
//                        print("Dự Đoán index: \(idx), score: \(score)")
//                        
//                        let label = (idx < self.labels.count) ? self.labels[idx] : "\(idx)"
//                        let text = String(format: "%@ (%.2f)", label, score)
//                        DispatchQueue.main.async {
//                            self.predictionLabel.text = text
//                            
//                            if let labelsPath = Bundle.main.path(forResource: "labels", ofType: "txt") {
//                                let labels = try? String(contentsOfFile: labelsPath).components(separatedBy: .newlines)
//                                
//                                
//                                if let maxIndex = results.firstIndex(of: results.max() ?? 0),
//                                   let label = labels?[maxIndex] {
//                                    print("👉 Dự đoán: \(label) (\(results[maxIndex]))")
//                                }
//                            }
//                            
//                            
//                        
//                        }
//                    } else {
//                        DispatchQueue.main.async {
//                            self.predictionLabel.text = "No result"
//                        }
//                    }
//                }
//                
//            } catch {
//                print("Lỗi khi chạy model: \(error)")
//                print("Failed to run interpreter: \(error)")
//            }
//        }
//    }
//    
//
//
//    override func viewDidLayoutSubviews() {
//        super.viewDidLayoutSubviews()
//        previewLayer.frame = previewView.bounds
//    }
//    
//    // Convert CVPixelBuffer to UIImage (simple)
//    private func uiImageFromPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> UIImage? {
//        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
//        let context = CIContext(options: nil)
//        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
//        return UIImage(cgImage: cgImage)
//    }
//}
//
//// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
//extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
//    func captureOutput(_ output: AVCaptureOutput,
//                       didOutput sampleBuffer: CMSampleBuffer,
//                       from connection: AVCaptureConnection) {
//        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
//        runModel(on: pixelBuffer)
//        
//    }
//    
//    // MARK: - Preprocess image
//    private func preprocess(pixelBuffer: CVPixelBuffer) -> Data? {
//        let width = 224
//        let height = 224
//        
//        // Resize ảnh về kích thước model yêu cầu
//        UIGraphicsBeginImageContext(CGSize(width: width, height: height))
//        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
//        let uiImage = UIImage(ciImage: ciImage)
//        uiImage.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
//        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
//        UIGraphicsEndImageContext()
//        
//        guard let rgbData = resizedImage?.normalizedData() else { return nil }
//        return rgbData
//    }
//}
