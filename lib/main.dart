import 'dart:isolate';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:mandelbrot/bindings/render_bindings.dart';
import 'package:texturerender/texturerender.dart';
import 'package:window_manager/window_manager.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;

late Texturerender tr;
final RenderFFI render = RenderFFI(ffi.DynamicLibrary.open("bin/render.dll"));

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
  int fWidth = 3840;
  int fHeight = 2160;
  Rect def = const Rect.fromLTRB(-2, -1, 1, 1);
  Rect rect = const Rect.fromLTRB(-2, -1, 1, 1);
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

    ffi.Pointer<ffi.Uint8> _pBuf; // is freed later by dispose

    _cReceivePort?.listen((data) {
      print(data);
      if (data is int) {
        _pBuf = ffi.Pointer.fromAddress(data);

        int s = fWidth * fHeight * 4;
        ffi.Pointer<ffi.Uint8> buffer = ffi.calloc.call<ffi.Uint8>(s);
        buffer.asTypedList(s).setAll(0, _pBuf.asTypedList(s));
        tr.update(texID, buffer, fWidth, fHeight);
        setState(() {
          rendering = false;
        });
        _cIsolate?.kill(priority: Isolate.immediate);
        _cIsolate = null;
        _cReceivePort?.close();
        _cReceivePort = null;
        render.dispose();
      }
    });

    setState(() {
      rendering = true;
    });

    _cIsolate = await Isolate.spawn(
        compute,
        CObject(
          _cReceivePort!.sendPort,
          fHeight: fHeight,
          fWidth: fWidth,
          l: rect.left,
          r: rect.right,
          t: rect.top,
          b: rect.bottom,
        ));
  }

  static void compute(CObject object) {
    ffi.Pointer<ffi.Uint8> pBuf =
        render.setupCanvas(object.fWidth, object.fHeight, object.l, object.r, object.b, object.t);

    for (int i = 0; i < 2000; i++) {
      Stopwatch sw = Stopwatch()..start();
      render.iterate();
      //print("iterating: ${sw.elapsedMilliseconds}");
    }
    render.draw();
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
    double real = rect.left + ((rect.right - rect.left) / s.width) * (pos.dx);
    double imag = rect.bottom + ((rect.top - rect.bottom) / s.height) * (s.height - pos.dy);

    return Offset(real, imag);
  }

  Offset? start;
  Offset? end;

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
                onPointerHover: (event) {
                  Offset pos = event.localPosition;
                },
                onPointerDown: (event) {
                  if (event.buttons & 4 != 0) {
                    setState(() {
                      prev = rect;
                      rect = def;
                      rerender();
                    });
                  }
                  if (event.buttons & 2 != 0 && prev != null) {
                    setState(() {
                      rect = prev!;
                      prev = null;
                      rerender();
                    });
                  }
                  if (event.buttons & 1 == 0) return;
                  Offset pos = event.localPosition;
                  Offset coords = posToCoord(pos, constraints.biggest);
                  start = coords;
                  print("(${coords.dx})+i(${coords.dy})");
                },
                onPointerUp: (event) {
                  Offset pos = event.localPosition;
                  Offset coords = posToCoord(pos, constraints.biggest);
                  end = coords;
                  if (start == null) return;

                  Offset topleft = Offset(
                    start!.dx < end!.dx ? start!.dx : end!.dx,
                    start!.dy > end!.dy ? start!.dy : end!.dy,
                  );
                  double w = (end!.dx - start!.dx).abs();
                  Size s = Size(w, -w / constraints.biggest.aspectRatio);
                  Rect rec = topleft & s;

                  print(rec);

                  setState(() {
                    prev = rect;
                    rect = rec;
                    rerender();
                  });
                  start = null;
                  end = null;
                },
                child: const MouseRegion(
                  cursor: SystemMouseCursors.precise,
                ),
              )
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
