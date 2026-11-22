import AVFoundation
import CoreVideo
import AppKit
import Metal
import SceneKit
import simd

struct Geometry: Decodable {
    let file: String
    let vertices: Int
    let triangles: Int
}
struct Part: Decodable {
    let body: Int
    let name: String
    let file: String
    let vertices: Int
    let triangles: Int
    var geometry: Geometry { Geometry(file: file, vertices: vertices, triangles: triangles) }
}
struct Track: Decodable {
    let folder, terrain, state: String
    let id, seed, update, bodies: Int
    let ticks, hold: [Int]
    let start, goal: [Float]
    let mesh: Geometry
}
let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let decoder = JSONDecoder()
func read<T: Decodable>(_ path: URL, _: T.Type) throws -> T {
    try decoder.decode(T.self, from: Data(contentsOf: path))
}
func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) -> NSColor {
    NSColor(srgbRed: r, green: g, blue: b, alpha: 1)
}
func material(_ tint: NSColor, shininess: CGFloat = 18) -> SCNMaterial {
    let m = SCNMaterial()
    m.diffuse.contents = tint
    m.lightingModel = .blinn
    m.specular.contents = color(0.14, 0.15, 0.17)
    m.shininess = shininess
    m.isDoubleSided = true
    return m
}
func geometry(_ spec: Geometry, in folder: URL, finish: SCNMaterial) throws -> SCNGeometry {
    let packed = try Data(contentsOf: folder.appendingPathComponent(spec.file))
    let verticesEnd = spec.vertices * 12
    guard packed.count == verticesEnd + spec.triangles * 12 else { fatalError("Bad geometry file \(spec.file)") }
    let vertices = packed.prefix(verticesEnd).withUnsafeBytes { buf -> [SIMD3<Float>] in
        let a = buf.bindMemory(to: Float.self)
        return (0..<spec.vertices).map { i in SIMD3<Float>(a[i*3], a[i*3+1], a[i*3+2]) }
    }
    let indices = packed.suffix(from: verticesEnd).withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
    var normals = Array(repeating: SIMD3<Float>(repeating: 0), count: spec.vertices)
    for t in stride(from: 0, to: indices.count, by: 3) {
        let a = Int(indices[t])
        let b = Int(indices[t+1])
        let c = Int(indices[t+2])
        let n = simd_cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        normals[a] += n
        normals[b] += n
        normals[c] += n
    }
    let normalVectors = normals.map { v -> SCNVector3 in
        let n = simd_length_squared(v) > 1e-12 ? simd_normalize(v) : SIMD3<Float>(0,0,1)
        return SCNVector3(n.x, n.y, n.z)
    }
    let points = SCNGeometrySource(vertices: vertices.map { SCNVector3($0.x, $0.y, $0.z) })
    let surfaceNormals = SCNGeometrySource(normals: normalVectors)
    let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
    let g = SCNGeometry(sources: [points, surfaceNormals], elements: [element])
    g.materials = [finish]
    return g
}
func lamp(_ type: SCNLight.LightType, intensity: CGFloat, tint: NSColor) -> SCNNode {
    let n = SCNNode()
    n.light = SCNLight()
    n.light!.type = type
    n.light!.intensity = intensity
    n.light!.color = tint
    return n
}
func vec(_ a: [Float]) -> SIMD3<Float> { SIMD3<Float>(a[0], a[1], a[2]) }

guard let metal = MTLCreateSystemDefaultDevice() else { fatalError("Metal device is unavailable") }
let renderer = SCNRenderer(device: metal, options: nil)
let parts = try read(root.appendingPathComponent("robot.json"), [Part].self)
let model = try parts.enumerated().map { ordinal, part -> SCNGeometry in
    let tint: NSColor
    if part.name.lowercased() == "base" {
        tint = [color(0.17, 0.21, 0.25), color(0.81, 0.85, 0.86), color(0.28, 0.34, 0.4)][ordinal % 3]
    } else if part.name.contains("FOOT") {
        tint = color(0.10, 0.14, 0.16)
    } else {
        tint = ordinal.isMultiple(of: 3) ? color(0.23, 0.27, 0.31) : color(0.72, 0.78, 0.81)
    }
    return try geometry(part.geometry, in: root, finish: material(tint, shininess: 26))
}

let prepared = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
let output = URL(fileURLWithPath: CommandLine.arguments[3], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let tracks = try read(prepared.appendingPathComponent("segments.json"), [Track].self)
let width = 960, height = 600
let fps: Int32 = 25
let space = CGColorSpace(name: CGColorSpace.sRGB)!
let rgba = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

func draw(_ string: String, x: CGFloat, y: CGFloat, size: CGFloat, weight: NSFont.Weight, color: NSColor, right: Bool = false) {
    let style: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: size, weight: weight), .foregroundColor: color]
    let label = NSAttributedString(string: string, attributes: style)
    label.draw(at: NSPoint(x: right ? x - label.size().width : x, y: y))
}


for track in tracks {
    let folder = prepared.appendingPathComponent(track.folder)
    let target = output.appendingPathComponent(track.folder+".mp4")
    let raw = try Data(contentsOf: folder.appendingPathComponent("poses.f32"))
    let samples = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    guard samples.count == track.ticks.count * track.bodies * 7 else { fatalError("Pose shape mismatch") }
let writer = try AVAssetWriter(outputURL: target, fileType: .mp4)
let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: width, AVVideoHeightKey: height,
    AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: 7_200_000, AVVideoExpectedSourceFrameRateKey: fps,
        AVVideoMaxKeyFrameIntervalKey: fps * 2
    ]
])
writerInput.expectsMediaDataInRealTime = false
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: writerInput, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
    kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
    kCVPixelBufferCGImageCompatibilityKey as String: true,
    kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
])
guard writer.canAdd(writerInput) else { fatalError("Video writer rejected its input") }
writer.add(writerInput)
guard writer.startWriting() else { fatalError(String(describing: writer.error)) }
writer.startSession(atSourceTime: .zero)
    let scene = SCNScene()
    let backdrop = color(0.50, 0.59, 0.66)
    scene.background.contents = backdrop
    let ground = try geometry(track.mesh, in: folder, finish: material(color(0.59, 0.64, 0.66), shininess: 1))
    scene.rootNode.addChildNode(SCNNode(geometry: ground))
    let surround = SCNPlane(width: 140, height: 140)
    surround.firstMaterial = material(color(0.33, 0.41, 0.47), shininess: 1)
    let below = SCNNode(geometry: surround)
    below.position.z = -1.85
    scene.rootNode.addChildNode(below)
    let bodies: [SCNNode] = (0..<track.bodies).map { _ in SCNNode() }
    for b in bodies { scene.rootNode.addChildNode(b) }
    for (p, g) in zip(parts, model) {
        bodies[p.body].addChildNode(SCNNode(geometry: g))
    }

    let goal = vec(track.goal)
    let pylon = SCNSphere(radius: 0.13)
    let marker = SCNNode(geometry: pylon)
    let orange = color(1.0, 0.34, 0.04)
    pylon.firstMaterial = material(orange)
    pylon.firstMaterial?.emission.contents = color(0.3, 0.065, 0.004)
    marker.simdPosition = goal + SIMD3<Float>(0, 0, 0.105)
    scene.rootNode.addChildNode(marker)

    scene.rootNode.addChildNode(lamp(.ambient, intensity: 460, tint: color(0.82, 0.88, 1.0)))
    let sun = lamp(.directional, intensity: 1100, tint: color(1.0, 0.94, 0.83))
    sun.simdPosition = SIMD3<Float>(4, -5, 9)
    sun.look(at: SCNVector3Zero, up: SCNVector3(0,0,1), localFront: SCNVector3(0,0,-1))
    sun.light!.castsShadow = true
    sun.light!.shadowMode = .forward
    sun.light!.shadowSampleCount = 12
    sun.light!.shadowRadius = 3
    sun.light!.shadowMapSize = CGSize(width: 1024, height: 1024)
    scene.rootNode.addChildNode(sun)
    let fill = lamp(.directional, intensity: 440, tint: color(0.76, 0.87, 1))
    fill.simdPosition = SIMD3<Float>(-4, 4, 5)
    fill.look(at: SCNVector3Zero, up: SCNVector3(0,0,1), localFront: SCNVector3(0,0,-1))
    scene.rootNode.addChildNode(fill)

    let camera = SCNNode()
    camera.camera = SCNCamera()
    camera.camera!.fieldOfView = 35
    camera.camera!.zFar = 100
    scene.rootNode.addChildNode(camera)
    renderer.scene = scene
    renderer.pointOfView = camera
    let start = vec(track.start)
    let along = simd_normalize(SIMD2<Float>(goal.x - start.x, goal.y - start.y))
    let sideways = SIMD2<Float>(-along.y, along.x) * 4.2 - along * 0.9
    var center = 0.72 * start + 0.28 * goal

    for (frame, tick) in track.ticks.enumerated() {
        try autoreleasepool {
            for (i, body) in bodies.enumerated() {
                let j = 7*(frame*track.bodies+i)
                body.simdPosition = SIMD3<Float>(samples[j],samples[j+1],samples[j+2])
                body.simdOrientation = simd_quatf(ix:samples[j+4],iy:samples[j+5],iz:samples[j+6],r:samples[j+3])
            }
            let base = bodies[0].simdPosition
            let focus = 0.72*base + 0.28*goal
            center = frame==0 ? focus : 0.80*center + 0.20*focus
            let spread = min(1.6,0.84 + 0.085*simd_distance(base, goal))
            camera.simdPosition = center + SIMD3<Float>(sideways.x*spread,sideways.y*spread,2.6*spread)
            camera.look(at: SCNVector3(center.x,center.y,center.z),up:SCNVector3(0,0,1),localFront:SCNVector3(0,0,-1))
            let picture=renderer.snapshot(atTime:Double(tick)/50,with:NSSize(width:width,height:height),antialiasingMode:.multisampling2X)
            var rect=CGRect(origin:.zero,size:picture.size)
            guard let image=picture.cgImage(forProposedRect:&rect,context:nil,hints:nil) else {throw CocoaError(.fileWriteUnknown)}
            var made: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(nil,adaptor.pixelBufferPool!,&made)==kCVReturnSuccess,let buffer=made else {throw CocoaError(.fileWriteUnknown)}
            CVPixelBufferLockBaseAddress(buffer,[])
            let ctx=CGContext(data:CVPixelBufferGetBaseAddress(buffer),width:width,height:height,bitsPerComponent:8,bytesPerRow:CVPixelBufferGetBytesPerRow(buffer),space:space,bitmapInfo:rgba)!
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(cgContext:ctx,flipped:false)
            ctx.draw(image,in:CGRect(x:0,y:0,width:width,height:height))
            ctx.setFillColor(NSColor.black.withAlphaComponent(0.68).cgColor)
            ctx.fill(CGRect(x:0,y:height-50,width:width,height:50))
            draw("\(track.terrain.replacingOccurrences(of:"_",with:" "))   ·   ID \(track.id)",x:16,y:CGFloat(height-34),size:22,weight:.semibold,color:.white)
            draw("seed \(track.seed)    u\(track.update)    "+String(format:"%.2f s",Double(tick)/50),x:CGFloat(width-15),y:CGFloat(height-30),size:17,weight:.regular,color:.white,right:true)
            ctx.fill(CGRect(x:0,y:0,width:width,height:34))
            draw("3 s hold   "+String(format:"%.2f / 3.00 s",Double(track.hold[frame])/50),x:16,y:7,size:16,weight:.medium,color:.white)
            draw(track.state.replacingOccurrences(of:"_",with:" "),x:CGFloat(width-15),y:7,size:16,weight:.medium,color:.white,right:true)
            NSGraphicsContext.restoreGraphicsState()
            CVPixelBufferUnlockBaseAddress(buffer,[])
            while !writerInput.isReadyForMoreMediaData {
                if writer.status == .failed {throw writer.error ?? CocoaError(.fileWriteUnknown)}
                Thread.sleep(forTimeInterval:0.002)
            }
            guard adaptor.append(buffer,withPresentationTime:CMTime(value:Int64(tick),timescale:50)) else {throw writer.error ?? CocoaError(.fileWriteUnknown)}
        }
    }
    writerInput.markAsFinished()
    writer.endSession(atSourceTime:CMTime(value:Int64(track.ticks.last!+1),timescale:50))
    let done=DispatchSemaphore(value:0)
    writer.finishWriting {done.signal()}
    done.wait()
    guard writer.status == .completed else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
    print(target.lastPathComponent)
}
