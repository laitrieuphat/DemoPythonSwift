////
////  UIImage.swift
////  DemoPythonSwift
////
////  Created by Lai Minh on 30/10/25.


import UIKit


// MARK: - UIImage extension for normalization
extension UIImage {
    func normalizedData() -> Data? {
        guard let cgImage = self.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerRow = width * 4
        var rawData = [UInt8](repeating: 0, count: Int(bytesPerRow * height))
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(data: &rawData, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                      space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var normalizedData = Data(capacity: width * height * 3 * MemoryLayout<Float32>.size)
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                let r = Float32(rawData[offset]) / 255.0
                let g = Float32(rawData[offset + 1]) / 255.0
                let b = Float32(rawData[offset + 2]) / 255.0
                var rgb: [Float32] = [r, g, b]
                normalizedData.append(UnsafeBufferPointer(start: &rgb, count: 3))
            }
        }
        return normalizedData
    }
}

extension Data {
    init<T>(fromArray values: [T]) {
        self = values.withUnsafeBufferPointer(Data.init)
    }
}
