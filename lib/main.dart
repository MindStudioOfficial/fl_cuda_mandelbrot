import 'dart:isolate';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:mandelbrot/bindings/render_bindings.dart';
import 'package:texturerender/texturerender.dart';
import 'package:window_manager/window_manager.dart';
import 'dart:ffi' as ffi;

late Texturerender tr;
final RenderFFI render = RenderFFI(ffi.DynamicLibrary.open("bin/render.dll"));

/*
You can change these parameters 
based on your preferences
*/
int frameWidth = 3840;
int frameHeight = 2160;
int iterations = 2000;

extension MapExtension on double {
  double map(double inMin, double inMax, double outMin, double outMax) {
    double slope = (outMax - outMin) / (inMax - inMin);
    return outMin + slope * (this - inMin);
  }
}

void main() async {
  tr = Texturerender();
  WidgetsFlutterBinding.ensureInitialized();
  await windowManager.ensureInitialized();
  WindowOptions wo = const WindowOptions(
    size: Size(2000, 1000),
    backgroundColor: Colors.transparent,
    titleBarStyle: TitleBarStyle.normal,
  );

  windowManager.waitUntilReadyToShow(wo, () async {
    await windowManager.show();
    await windowManager.focus();
  });
  runApp(const Main());
}

class Main extends StatefulWidget {
  const Main({super.key});

  @override
  State<Main> createState() => _MainState();
}

class _MainState extends State<Main> with WindowListener {
  int texID = 1;
  bool texInit = false;

  Rect def = const Rect.fromLTRB(-2, -1, 1, 1);
  Rect complexPlaneRect = const Rect.fromLTRB(-2, -1, 1, 1);
  Rect? prev;

  bool rendering = false;

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    windowManager.setPreventClose(true).then((value) => setState(() {}));
    WidgetsFlutterBinding.ensureInitialized().addPostFrameCallback((timeStamp) {
      init();
    });
  }

  Future<void> init() async {
    await tr.register(texID);
    setState(() {
      texInit = true;
    });

    rerender();
  }

  Future<void> rerender() async {
    _cReceivePort = ReceivePort();

    ffi.Pointer<ffi.Uint8> pBuf;

    _cReceivePort?.listen((data) {
      if (data is int) {
        pBuf = ffi.Pointer.fromAddress(data);

        tr.update(texID, pBuf, frameWidth, frameHeight);
        setState(() {
          rendering = false;
        });
        _cIsolate?.kill(priority: Isolate.immediate);
        _cIsolate = null;
        _cReceivePort?.close();
        _cReceivePort = null;
      }
    });

    setState(() {
      rendering = true;
    });

    _cIsolate = await Isolate.spawn(
        compute,
        CObject(
          _cReceivePort!.sendPort,
          fHeight: frameHeight,
          fWidth: frameWidth,
          l: complexPlaneRect.left,
          r: complexPlaneRect.right,
          t: complexPlaneRect.top,
          b: complexPlaneRect.bottom,
        ));
  }

  static void compute(CObject object) {
    Stopwatch sw = Stopwatch()..start();
    ffi.Pointer<ffi.Uint8> pBuf =
        render.setupCanvas(object.fWidth, object.fHeight, object.l, object.r, object.t, object.b);

    render.iterate(iterations);

    if (kDebugMode) {
      print("Finished iterating after ${sw.elapsedMicroseconds}μs");
    }
    render.draw();
    if (kDebugMode) {
      print("Finished rendering after ${sw.elapsedMicroseconds}μs");
    }
    render.dispose();
    object.sp.send(pBuf.address);
  }

  Isolate? _cIsolate;
  ReceivePort? _cReceivePort;

  @override
  void dispose() async {
    await tr.dispose();
    windowManager.removeListener(this);
    super.dispose();
  }

  @override
  void onWindowClose() async {
    super.onWindowClose();
    setState(() {
      texInit = false;
    });
    await tr.dispose();
    await windowManager.destroy();
  }

  Offset posToCoord(Offset pos, Size s) {
    double real = pos.dx.map(0, s.width, complexPlaneRect.left, complexPlaneRect.right);
    double imag = pos.dy.map(0, s.height, complexPlaneRect.bottom, complexPlaneRect.top);

    return Offset(real, imag);
  }

  Offset coordToPos(Offset coord, Rect rect, Size size) {
    double x = coord.dx.map(rect.left, rect.right, 0, size.width);
    double y = coord.dy.map(rect.bottom, rect.top, 0, size.height);
    return Offset(x, y);
  }

  Offset? startCoords;
  Offset? endCoords;
  Offset? current;
  bool dragging = false;
  Rect? currentRect;
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        body: LayoutBuilder(builder: (context, constraints) {
          return Stack(
            fit: StackFit.expand,
            children: [
              if (texInit) tr.widget(texID),
              if (!texInit) const Placeholder(),
              if (rendering)
                const Center(
                  child: CircularProgressIndicator(),
                ),
              Listener(
                onPointerMove: (event) {
                  //* UPDATE CURRENT
                  Offset pos = event.localPosition;
                  setState(() {
                    current = posToCoord(pos, constraints.biggest);
                  });

                  //* GET START POSITION FOR VISUAL RECT
                  if (startCoords == null) return;

                  Offset startPos = coordToPos(startCoords!, complexPlaneRect, constraints.biggest);

                  Rect t = Rect.fromPoints(startPos, pos);

                  double hDelta = t.height - t.width / constraints.biggest.aspectRatio;

                  if (pos.dy > startPos.dy) pos = Offset(pos.dx, pos.dy - hDelta);
                  if (pos.dy <= startPos.dy) pos = Offset(pos.dx, pos.dy + hDelta);

                  t = Rect.fromPoints(startPos, pos);

                  setState(() {
                    currentRect = t;
                  });
                },
                onPointerHover: (event) {
                  Offset pos = event.localPosition;
                  setState(() {
                    current = posToCoord(pos, constraints.biggest);
                  });
                },
                onPointerDown: (event) {
                  if (event.buttons & 4 != 0) {
                    // MIDDLE MOUS BUTTON = RESET
                    setState(() {
                      prev = complexPlaneRect;
                      complexPlaneRect = def;
                      rerender();
                    });
                  }
                  if (event.buttons & 2 != 0 && prev != null) {
                    // RIGHT MOUSE BUTTON = PREVIOUS
                    setState(() {
                      complexPlaneRect = prev!;
                      prev = null;
                      rerender();
                    });
                  }
                  if (event.buttons & 1 == 0) return; // LEFT MOUSE BUTTON = start draw rect
                  dragging = true;
                  Offset pos = event.localPosition;
                  Offset coords = posToCoord(pos, constraints.biggest);
                  setState(() {
                    startCoords = coords;
                  });
                  if (kDebugMode) {
                    print("(${coords.dx})+i(${coords.dy})");
                  }
                },
                onPointerUp: (event) {
                  if (!dragging) return;
                  dragging = false;
                  if (currentRect == null) return;

                  setState(() {
                    complexPlaneRect = Rect.fromPoints(
                      posToCoord(currentRect!.topLeft, constraints.biggest),
                      posToCoord(currentRect!.bottomRight, constraints.biggest),
                    );
                    prev = complexPlaneRect;
                    currentRect = null;
                    rerender();
                  });
                },
                child: const MouseRegion(
                  cursor: SystemMouseCursors.precise,
                ),
              ),
              Align(
                alignment: Alignment.topLeft,
                child: Text(
                  "${current?.dx} + ${current?.dy}i",
                  //"Real: ${complexPlaneRect.left} to ${complexPlaneRect.right}"
                  //"Imag: ${complexPlaneRect.top} to ${complexPlaneRect.bottom}"

                  style: const TextStyle(color: Colors.white, fontSize: 25),
                ),
              ),
              if (currentRect != null && dragging)
                Positioned(
                  top: currentRect!.top,
                  left: currentRect!.left,
                  child: Container(
                    color: Colors.white.withOpacity(.5),
                    width: currentRect!.width,
                    height: currentRect!.height,
                  ),
                ),
            ],
          );
        }),
      ),
    );
  }
}

class CObject {
  SendPort sp;
  int fWidth;
  int fHeight;
  double l, t, b, r;
  CObject(
    this.sp, {
    required this.fHeight,
    required this.fWidth,
    required this.l,
    required this.b,
    required this.r,
    required this.t,
  });
}
