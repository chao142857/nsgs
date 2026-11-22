import CoreVideo
import AVFoundation
import Foundation
struct Segment: Decodable {
    let folder: String
    let ticks: [Int]
}
let dir = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let list = try JSONDecoder().decode([Segment].self,from:Data(contentsOf:dir.appendingPathComponent("prepared/segments.json")))
for segment in list {
    let file = dir.appendingPathComponent(segment.folder+".mp4")
    let asset=AVURLAsset(url:file)
    guard let track=asset.tracks(withMediaType:.video).first else {fatalError(file.lastPathComponent)}
    let expected=Double(segment.ticks.last!+1)/50.0
    let actual=asset.duration.seconds
    guard abs(actual-expected)<=0.045,track.naturalSize.width==960,track.naturalSize.height==600,
          segment.ticks.first==0 else {fatalError("\(file.lastPathComponent) expected \(expected), got \(actual)")}
    let reader=try AVAssetReader(asset:asset)
    let output=AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)])
    reader.add(output)
    guard reader.startReading() else {fatalError("Cannot read \(file.lastPathComponent)")}
    var frame=0
    while let packet=output.copyNextSampleBuffer() {
        let tick=CMSampleBufferGetPresentationTimeStamp(packet).seconds*50
        guard frame<segment.ticks.count,abs(tick-Double(segment.ticks[frame]))<=0.06 else {
            fatalError("Frame clock mismatch \(file.lastPathComponent) at \(frame), actual \(tick)")
        }
        frame+=1
    }
    guard frame==segment.ticks.count else {fatalError("Frame count mismatch \(file.lastPathComponent)")}
    print(segment.folder,frame,actual)
}
