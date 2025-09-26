#!/usr/bin/env python3
# nest_dxf_qty_optimized_shuffle.py — Windows-only, no installs
# - Reads DXF loops (LINE/LWPOLYLINE/POLYLINE/ARC/CIRCLE); joins open segments
# - Uses same-named .txt files to pull Quantity (CSV: Part,Configuration,Quantity)
# - Bitmap nesting with exact spacing and a 1px safety pad (prevents aliasing leaks)
# - Multi-try shuffle to improve packing (fewest sheets, then densest fill)
# - Optional: forbid parts being placed inside other parts' holes
# - Draws one border per sheet with configurable margin + gap; saves nested.dxf + nest_report.txt
# - Shows a tiny progress window (Win32 via ctypes) — no tkinter required

# ======= SETTINGS =======
# NOTE: The folder defaults to the sample DXFs that ship with the repository when
# available.  On Windows the original absolute path is left as a fallback so the
# script behaves the same when copied back to its source environment.
FOLDER = r"C:\Users\Jsudhakaran\OneDrive - GN Corporation Inc\Desktop\test\For waterjet cutting"

# Sheet (inches)
SHEET_W = 48.0
SHEET_H = 96.0
SHEET_MARGIN = 0.50     # empty border inside the sheet frame
SHEET_GAP = 2.0         # gap between sheet frames in the output DXF

# Packing / spacing
SPACING  = 0.10         # desired edge-to-edge clearance between parts (same units as DXF)

# Parser / geometry tolerances
JOIN_TOL = 0.005        # endpoint snap tolerance when joining segments into loops
ARC_CHORD_TOL = 0.01    # arc flattening chord tolerance (smaller = more segments)

# Behaviors
FALLBACK_OPEN_AS_BBOX = True   # if no closed loops, use overall DXF bbox as a rectangle

ALLOW_ROTATE_90       = True
ALLOW_MIRROR          = False  # allow mirrored placements (flip across Y axis)
USE_OBB_CANDIDATE     = True   # try oriented bounding box angles too

INSUNITS = 1                   # 1=inches, 4=mm (stored in DXF header; advisory only)

# Rectangle orientation preference: "off" | "prefer" | "force"
RECT_ALIGN_MODE = "prefer"
RECT_ALIGN_TOL  = 1e-3

# Hole nesting policy:
ALLOW_NEST_IN_HOLES = True     # True: allow part-in-hole. False: holes are blocked areas.

# Nesting engine
NEST_MODE = "bitmap"           # "bitmap" | "shelf" (shelf = simpler fallback)

PIXELS_PER_UNIT = 20           # ↑ = tighter/more accurate (slower)

# Worker processes used by the bitmap evaluator.  Leaving this at ``None``
# lets the script auto-detect the CPU count once ``os`` is available.
BITMAP_EVAL_WORKERS = None

# Optional PyTorch device string for the bitmap accelerator ("cuda", "cuda:0", "cpu", etc.).
BITMAP_DEVICE = None  # type: Optional[str]

# Worker processes used by the bitmap evaluator.  Leaving this at ``None``
# lets the script auto-detect the CPU count once ``os`` is available.
BITMAP_EVAL_WORKERS = None

# Optional PyTorch device string for the bitmap accelerator ("cuda", "cuda:0", "cpu", etc.).
BITMAP_DEVICE = None  # type: Optional[str]

# Worker processes used by the bitmap evaluator.  Leaving this at ``None``
# lets the script auto-detect the CPU count once ``os`` is available.
BITMAP_EVAL_WORKERS = None

# Optional PyTorch device string for the bitmap accelerator ("cuda", "cuda:0", "cpu", etc.).
BITMAP_DEVICE = None  # type: Optional[str]

# Worker processes used by the bitmap evaluator.  Leaving this at ``None``
# lets the script auto-detect the CPU count once ``os`` is available.
BITMAP_EVAL_WORKERS = None



# Multi-try randomization (bitmap only)
SHUFFLE_TRIES = 5

SHUFFLE_SEED  = None           # int for reproducibility, or None
# ========================

import os, math

from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import Random

# Detect a co-located sample folder so out-of-the-box runs on Linux/macOS pick
# up the repository assets without having to edit the script manually.

_REPO_SAMPLE_FOLDER = os.path.join(os.path.dirname(__file__), "For waterjet cutting")
if os.path.isdir(_REPO_SAMPLE_FOLDER):
    FOLDER = _REPO_SAMPLE_FOLDER




if not BITMAP_EVAL_WORKERS:
    cpu_count = os.cpu_count() or 1
    BITMAP_EVAL_WORKERS = max(1, cpu_count)

# ---------- tiny Windows progress window (robust prototypes) ----------
IS_WINDOWS = (os.name == "nt")
if IS_WINDOWS:
    import ctypes
    user32  = ctypes.windll.user32
    gdi32   = ctypes.windll.gdi32
    kernel32= ctypes.windll.kernel32

    # Basic C/Win types
    UINT    = ctypes.c_uint
    DWORD   = ctypes.c_uint
    INT     = ctypes.c_int
    LONG    = ctypes.c_long
    ULONG_PTR = ctypes.c_size_t
    LONG_PTR  = ctypes.c_ssize_t
    WPARAM = ULONG_PTR
    LPARAM = LONG_PTR
    LRESULT = LONG_PTR

    HWND    = ctypes.c_void_p
    HINSTANCE = ctypes.c_void_p
    HICON   = ctypes.c_void_p
    HCURSOR = ctypes.c_void_p
    HBRUSH  = ctypes.c_void_p
    HMENU   = ctypes.c_void_p
    LPCWSTR = ctypes.c_wchar_p

    # Win constants
    WS_OVERLAPPEDWINDOW = 0x00CF0000
    WS_VISIBLE = 0x10000000
    WS_CHILD   = 0x40000000
    WS_EX_TOPMOST = 0x00000008
    SW_SHOWNORMAL = 1
    WM_DESTROY = 0x0002
    WM_CLOSE   = 0x0010
    PM_REMOVE  = 0x0001

    SS_LEFT = 0x00000000
    SS_NOPREFIX = 0x00000080

    WHITE_BRUSH = 0  # GetStockObject(WHITE_BRUSH)

    class POINT(ctypes.Structure):
        _fields_ = [("x", LONG), ("y", LONG)]

    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, HWND, UINT, WPARAM, LPARAM)

    class WNDCLASS(ctypes.Structure):
        _fields_ = [("style", UINT),
                    ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", INT),
                    ("cbWndExtra", INT),
                    ("hInstance", HINSTANCE),
                    ("hIcon", HICON),
                    ("hCursor", HCURSOR),
                    ("hbrBackground", HBRUSH),
                    ("lpszMenuName", LPCWSTR),
                    ("lpszClassName", LPCWSTR)]

    class MSG(ctypes.Structure):
        _fields_ = [("hwnd", HWND),
                    ("message", UINT),
                    ("wParam", WPARAM),
                    ("lParam", LPARAM),
                    ("time", DWORD),
                    ("pt", POINT)]

    # Correct prototypes
    user32.DefWindowProcW.argtypes = [HWND, UINT, WPARAM, LPARAM]
    user32.DefWindowProcW.restype  = LRESULT
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
    user32.RegisterClassW.restype  = ctypes.c_ushort
    user32.CreateWindowExW.argtypes = [DWORD, LPCWSTR, LPCWSTR, DWORD, INT, INT, INT, INT, HWND, HMENU, HINSTANCE, ctypes.c_void_p]
    user32.CreateWindowExW.restype = HWND
    user32.DestroyWindow.argtypes = [HWND]
    user32.DestroyWindow.restype  = ctypes.c_int
    user32.ShowWindow.argtypes    = [HWND, INT]
    user32.UpdateWindow.argtypes  = [HWND]
    user32.SetWindowTextW.argtypes= [HWND, LPCWSTR]
    user32.SetWindowTextW.restype = ctypes.c_int
    user32.GetSystemMetrics.argtypes = [INT]
    user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), HWND, UINT, UINT, UINT]
    user32.PeekMessageW.restype  = ctypes.c_int
    user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
    user32.DispatchMessageW.restype  = LRESULT
    user32.PostQuitMessage.argtypes = [INT]
    gdi32.GetStockObject.argtypes = [INT]
    gdi32.GetStockObject.restype  = HBRUSH

    DefWindowProcW = user32.DefWindowProcW

    class WinProgress:
        def __init__(self, title="Nesting DXF…", width=480, height=220):
            self.enabled = True
            self.title = title
            self.width = width
            self.height = height
            self.hInstance = kernel32.GetModuleHandleW(None)
            self.hwnd = HWND()
            self.hStatic = HWND()
            self._wndproc = None

        def create(self):
            try:
                @WNDPROC
                def wndproc(hwnd, msg, wParam, lParam):
                    if msg == WM_DESTROY:
                        user32.PostQuitMessage(0)
                        return LRESULT(0)
                    try:
                        return DefWindowProcW(hwnd, msg, wParam, lParam)
                    except Exception:
                        return LRESULT(0)
                self._wndproc = wndproc

                cls = WNDCLASS()
                cls.style = 0
                cls.lpfnWndProc = self._wndproc
                cls.cbClsExtra = 0
                cls.cbWndExtra = 0
                cls.hInstance = self.hInstance
                cls.hIcon = HICON()
                cls.hCursor = HCURSOR()
                cls.hbrBackground = gdi32.GetStockObject(WHITE_BRUSH)
                cls.lpszMenuName = None
                cls.lpszClassName = "PyNestProgress"
                try:
                    user32.RegisterClassW(ctypes.byref(cls))
                except Exception:
                    pass

                sw = user32.GetSystemMetrics(0)
                sh = user32.GetSystemMetrics(1)
                x = max(0, (sw - self.width)//2)
                y = max(0, (sh - self.height)//2)

                self.hwnd = user32.CreateWindowExW(
                    WS_EX_TOPMOST,
                    "PyNestProgress",
                    self.title,
                    WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                    x, y, self.width, self.height,
                    None, None, self.hInstance, None
                )
                if not self.hwnd:
                    self.enabled = False
                    return

                self.hStatic = user32.CreateWindowExW(
                    0, "STATIC", "Loading…",
                    WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOPREFIX,
                    12, 12, self.width-24, self.height-24,
                    self.hwnd, None, self.hInstance, None
                )

                user32.ShowWindow(self.hwnd, SW_SHOWNORMAL)
                user32.UpdateWindow(self.hwnd)
                self.pump()
            except Exception:
                self.enabled = False

        def pump(self):
            if not self.enabled: return
            msg = MSG()
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

        def update(self, text: str):
            if not self.enabled: return
            try:
                user32.SetWindowTextW(self.hStatic, text)
                self.pump()
            except Exception:
                pass

        def close(self):
            if not self.enabled: return
            try:
                user32.DestroyWindow(self.hwnd)
                self.hwnd = HWND()
            except Exception:
                pass
else:
    class WinProgress:
        def __init__(self, *_, **__): self.enabled=False
        def create(self): pass
        def update(self, _): pass
        def pump(self): pass
        def close(self): pass

# --------- simple logger ----------
_report_lines: List[str] = []
def log(line: str):
    print(line)
    _report_lines.append(line)


def _write_report(folder: str, extra_lines: Optional[List[str]] = None) -> Optional[str]:
    """Persist the accumulated log/report lines.

    The report normally lives alongside the DXFs, but if that directory is
    unavailable we fall back to the script directory so errors are still
    captured when the script exits early.
    """

    target_dir = folder if folder and os.path.isdir(folder) else os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(target_dir, "nest_report.txt")

    payload: List[str] = ["=== Nesting Report ==="]
    payload.extend(_report_lines)
    if extra_lines:
        if payload and payload[-1] != "":
            payload.append("")
        payload.extend(extra_lines)

    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write("\n".join(payload) + "\n")
    except Exception as exc:
        print(f"[WARN] Could not write report: {exc}")
        return None

    return report_path


def _write_report(folder: str, extra_lines: Optional[List[str]] = None) -> Optional[str]:
    """Persist the accumulated log/report lines.

    The report normally lives alongside the DXFs, but if that directory is
    unavailable we fall back to the script directory so errors are still
    captured when the script exits early.
    """

    target_dir = folder if folder and os.path.isdir(folder) else os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(target_dir, "nest_report.txt")

    payload: List[str] = ["=== Nesting Report ==="]
    payload.extend(_report_lines)
    if extra_lines:
        if payload and payload[-1] != "":
            payload.append("")
        payload.extend(extra_lines)

    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write("\n".join(payload) + "\n")
    except Exception as exc:
        print(f"[WARN] Could not write report: {exc}")
        return None

    return report_path

Point = Tuple[float,float]
Loop  = List[Point]
Seg   = Tuple[Point,Point]

# ---------- Qty ----------
def read_qty_for_dxf(folder: str, dxf_filename: str) -> int:
    base, _ = os.path.splitext(dxf_filename)
    for ext in ('.txt', '.TXT'):
        p = os.path.join(folder, base + ext)
        if os.path.isfile(p):
            try:
                lines = [ln.strip() for ln in open(p,'r',encoding='utf-8',errors='ignore') if ln.strip()]
                if not lines: return 1
                start = 1 if ('quantity' in lines[0].lower()) else 0
                total = 0
                for ln in lines[start:]:
                    cells = [c.strip() for c in ln.split(',')]
                    token = cells[-1] if cells else ln
                    q = None
                    try:
                        q = int(float(token))
                    except:
                        digs=''
                        for ch in ln:
                            if ch.isdigit(): digs+=ch
                            elif digs: break
                        if digs: q = int(digs)
                    if q and q>0: total += q
                return total if total>0 else 1
            except Exception:
                return 1
    return 1

# ---------- DXF parsing / joining ----------
def _read_text(p):
    with open(p,'r',encoding='utf-8',errors='ignore') as f:
        return f.read().splitlines()

def _arc_points(cx,cy,r,a0_deg,a1_deg,chord_tol):
    a0=math.radians(a0_deg); a1=math.radians(a1_deg)
    while a1<a0: a1+=2*math.pi
    sweep=a1-a0
    if r<=0: return [(cx,cy)]
    dtheta=2*math.asin(max(0.0,min(1.0,chord_tol/(2*r)))) if chord_tol>0 else (math.pi/36)
    steps=max(2,int(math.ceil(sweep/max(dtheta,1e-6))))
    return [(cx+r*math.cos(a0+sweep*k/steps), cy+r*math.sin(a0+sweep*k/steps)) for k in range(steps+1)]

def parse_entities(path: str):
    lines=_read_text(path)
    loops=[]; segs=[]
    in_entities=False
    in_lw=False; lw_pts=[]; lw_closed=False
    in_poly=False; poly_pts=[]; poly_closed=False
    i=0; n=len(lines)
    def get(i): return lines[i].strip(), lines[i+1].strip()
    while i+1<n:
        code,val=get(i); i+=2
        if code=='0' and val=='SECTION':
            if i+1<n:
                c2,v2=get(i)
                if c2=='2' and v2=='ENTITIES': in_entities=True
            continue
        if code=='0' and val=='ENDSEC': in_entities=False; continue
        if not in_entities: continue
        if code=='0':
            if in_lw:
                if lw_pts:
                    if lw_closed and lw_pts[0]!=lw_pts[-1]: lw_pts.append(lw_pts[0])
                    if len(lw_pts)>=4: loops.append(lw_pts)
                in_lw=False; lw_pts=[]; lw_closed=False
            if in_poly:
                if poly_pts:
                    if poly_closed and poly_pts[0]!=poly_pts[-1]: poly_pts.append(poly_pts[0])
                    if len(poly_pts)>=4: loops.append(poly_pts)
                in_poly=False; poly_pts=[]; poly_closed=False
            if val=='LWPOLYLINE': in_lw=True; continue
            if val=='POLYLINE':   in_poly=True; continue
            if val=='LINE':
                x1=y1=x2=y2=None
                while i+1<n:
                    c3,v3=get(i); i+=2
                    if c3=='0': i-=2; break
                    if c3=='10': x1=float(v3)
                    elif c3=='20': y1=float(v3)
                    elif c3=='11': x2=float(v3)
                    elif c3=='21': y2=float(v3)
                if None not in (x1,y1,x2,y2): segs.append(((x1,y1),(x2,y2)))
                continue
            if val in ('ARC','CIRCLE'):
                cx=cy=r=None
                a0 = 0.0 if val=='CIRCLE' else None
                a1 = 360.0 if val=='CIRCLE' else None
                while i+1<n:
                    c3,v3=get(i); i+=2
                    if c3=='0': i-=2; break
                    if   c3=='10': cx=float(v3)
                    elif c3=='20': cy=float(v3)
                    elif c3=='40': r =float(v3)
                    elif c3=='50': a0=float(v3)
                    elif c3=='51': a1=float(v3)
                if None not in (cx,cy,r,a0,a1):
                    pts=_arc_points(cx,cy,r,a0,a1,ARC_CHORD_TOL)
                    for k in range(len(pts)-1):
                        segs.append((pts[k],pts[k+1]))
                continue
            continue
        if in_lw:
            if code=='10':
                x=float(val)
                if i+1<n:
                    c2,v2=get(i); i+=2
                    if c2=='20': lw_pts.append((x,float(v2)))
                    else: i-=2
            elif code=='70':
                try: flags=int(val)
                except: flags=0
                lw_closed=bool(flags&1)
        elif in_poly:
            if code=='70':
                try: flags=int(val)
                except: flags=0
                poly_closed=bool(flags&1)
            elif code=='10':
                x=float(val)
                if i+1<n:
                    c2,v2=get(i); i+=2
                    if c2=='20': poly_pts.append((x,float(v2)))
                    else: i-=2

    if in_lw and lw_pts:
        if lw_closed and lw_pts[0]!=lw_pts[-1]: lw_pts.append(lw_pts[0])
        if len(lw_pts)>=4: loops.append(lw_pts)
    if in_poly and poly_pts:
        if poly_closed and poly_pts[0]!=poly_pts[-1]: poly_pts.append(poly_pts[0])
        if len(poly_pts)>=4: loops.append(poly_pts)
    return loops, segs

def join_segments_to_loops(segs: List[Seg], tol=JOIN_TOL) -> List[Loop]:
    if not segs: return []
    def key(pt): return (round(pt[0]/tol), round(pt[1]/tol))
    adj: Dict[tuple,List[tuple]]={}
    used=[False]*len(segs)
    for idx,(a,b) in enumerate(segs):
        ka,kb=key(a),key(b)
        adj.setdefault(ka,[]).append((a,b,idx))
        adj.setdefault(kb,[]).append((b,a,idx))
    loops=[]
    for idx,(a0,b0) in enumerate(segs):
        if used[idx]: continue
        chain=[a0,b0]; used[idx]=True
        # forward
        end=b0; kend=key(end)
        while True:
            nxt=None
            for a,b,j in adj.get(kend,[]):
                if used[j]: continue
                if abs(a[0]-end[0])<=tol and abs(a[1]-end[1])<=tol:
                    nxt=(b,j); break
            if not nxt: break
            chain.append(nxt[0]); used[nxt[1]]=True; end=nxt[0]; kend=key(end)
        # backward
        start=a0; kstart=key(start)
        while True:
            prv=None
            for a,b,j in adj.get(kstart,[]):
                if used[j]: continue
                if abs(b[0]-start[0])<=tol and abs(b[1]-start[1])<=tol:
                    prv=(a,j); break
            if not prv: break
            chain.insert(0,prv[0]); used[prv[1]]=True; start=prv[0]; kstart=key(start)
        if len(chain)>=4 and abs(chain[0][0]-chain[-1][0])<=tol and abs(chain[0][1]-chain[-1][1])<=tol:
            if chain[0]!=chain[-1]: chain.append(chain[0])
            loops.append(chain)
    return loops

# ---------- geometry ----------
def polygon_area(loop: Loop) -> float:
    s=0.0
    for i in range(len(loop)-1):
        x1,y1=loop[i]; x2,y2=loop[i+1]
        s += x1*y2 - x2*y1
    return 0.5*s

def bbox_of_points(pts: List[Point]):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return min(xs),min(ys),max(xs),max(ys)

def bbox_of_loops(loops: List[Loop]):
    pts=[p for lp in loops for p in lp]
    return bbox_of_points(pts) if pts else (0,0,0,0)


def translate_loop(loop: Loop, dx: float, dy: float) -> Loop:
    return [(x+dx,y+dy) for x,y in loop]

def mirror_loop(loop: Loop) -> Loop:
    mirrored = [(-x, y) for x, y in loop]
    minx = min((x for x, _ in mirrored), default=0.0)
    miny = min((y for _, y in mirrored), default=0.0)
    return [(x - minx, y - miny) for x, y in mirrored]

def rotate_loop(loop: Loop, theta: float) -> Loop:
    c,s=math.cos(theta), math.sin(theta)
    rot=[(x*c - y*s, x*s + y*c) for x,y in loop]
    minx=min(x for x,_ in rot); miny=min(y for _,y in rot)
    return [(x-minx,y-miny) for x,y in rot]


def convex_hull(points: List[Point]) -> List[Point]:
    pts=sorted(set(points))
    if len(pts)<=1: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def min_area_rect(points: List[Point]):
    hull=convex_hull(points)
    if len(hull)<=1: return 0.0,0.0,0.0
    best=(float('inf'),0.0,0.0,0.0)
    for i in range(len(hull)):
        x1,y1=hull[i]; x2,y2=hull[(i+1)%len(hull)]
        theta=math.atan2(y2-y1, x2-x1)
        ct,st=math.cos(-theta), math.sin(-theta)
        xs=[px*ct - py*st for px,py in hull]
        ys=[px*st + py*ct for px,py in hull]
        w=max(xs)-min(xs); h=max(ys)-min(ys); area=w*h
        if area<best[0]: best=(area,w,h,theta)
    _,w,h,theta=best
    return w,h,theta

def is_rect_like_by_area(outer_loop, obb_w, obb_h, tol_frac=RECT_ALIGN_TOL) -> bool:
    rect_area = obb_w * obb_h
    if rect_area <= 0: return False
    poly_area = abs(polygon_area(outer_loop))
    return abs(poly_area - rect_area) <= tol_frac * rect_area

def split_outer_and_holes(loops: List[Loop]):
    if not loops: return None,[]
    idx=max(range(len(loops)), key=lambda i: abs(polygon_area(loops[i])))
    return loops[idx], [loops[i] for i in range(len(loops)) if i!=idx]

# ---------- Part ----------

class Part:
    _uid_counter = 0

    def __init__(self, name: str, loops: List[Loop], fallback_bbox: Optional[Tuple[float,float,float,float]]):
        if loops:
            minx,miny,maxx,maxy=bbox_of_loops(loops)
            loops0=[translate_loop(lp,-minx,-miny) for lp in loops]
        elif fallback_bbox is not None:

            minx,miny,maxx,maxy=fallback_bbox
            loops0=[[ (0,0),(maxx-minx,0),(maxx-minx,maxy-miny),(0,maxy-miny),(0,0) ]]
        else:
            loops0=[]

        self.name=name
        if not loops0:
            self.outer=None; self.holes=[]; self.w=self.h=0.0; self.obb_w=self.obb_h=self.obb_theta=0.0; return
        self.outer,self.holes = split_outer_and_holes(loops0)
        minx,miny,maxx,maxy=bbox_of_loops([self.outer])
        self.w=maxx-minx; self.h=maxy-miny
        self.obb_w,self.obb_h,self.obb_theta = min_area_rect(self.outer)

        self._cand_cache = {}  # (scale, angle) -> dict(loops, raw, test, shell, pw,ph)
        self.uid = Part._uid_counter
        Part._uid_counter += 1

    def oriented(self, theta: float):
        if self.outer is None: return 0.0,0.0,[]
        if abs(theta)%(2*math.pi) < 1e-12: return self.w,self.h,[self.outer]+self.holes
        loops_r=[rotate_loop(lp, theta) for lp in [self.outer]+self.holes]
        minx,miny,maxx,maxy=bbox_of_loops([loops_r[0]])
        return (maxx-minx),(maxy-miny),loops_r



    def _axis_align_angles(self):
        a = (-self.obb_theta) % math.pi
        return [a, (a + math.pi/2) % math.pi]

    def is_rect_like(self) -> bool:
        return self.outer is not None and is_rect_like_by_area(self.outer, self.obb_w, self.obb_h)

    def candidate_angles(self):
        base = [0.0]
        if ALLOW_ROTATE_90: base.append(math.pi/2)
        if USE_OBB_CANDIDATE and self.obb_w>0 and self.obb_h>0:
            a = self.obb_theta % math.pi
            base += [a, (a + math.pi/2) % math.pi]
        if RECT_ALIGN_MODE in ("prefer","force") and self.is_rect_like():
            axis = self._axis_align_angles()
            angs = axis if RECT_ALIGN_MODE=="force" else (axis + base)
        else:
            angs = base
        out=[]
        for a in angs:
            if all(abs((a-b)%(math.pi))>math.radians(1) for b in out):
                out.append(a)
        return out

    def clone_for_worker(self) -> 'Part':
        clone = Part.__new__(Part)
        clone.name = self.name
        clone.outer = self.outer
        clone.holes = self.holes
        clone.w = self.w
        clone.h = self.h
        clone.obb_w = self.obb_w
        clone.obb_h = self.obb_h
        clone.obb_theta = self.obb_theta
        clone._cand_cache = {}
        clone.uid = self.uid
        return clone

    def candidate_poses(self):
        angles = self.candidate_angles()
        mirrors = [False, True] if ALLOW_MIRROR else [False]
        seen = set()
        poses = []
        for mirror in mirrors:
            for ang in angles:
                key = (mirror, round((ang % (2*math.pi)), 10))
                if key in seen:
                    continue
                seen.add(key)
                poses.append((ang, mirror))
        return poses

    def candidate_poses(self):
        angles = self.candidate_angles()
        mirrors = [False, True] if ALLOW_MIRROR else [False]
        seen = set()
        poses = []
        for mirror in mirrors:
            for ang in angles:
                key = (mirror, round((ang % (2*math.pi)), 10))
                if key in seen:
                    continue
                seen.add(key)
                poses.append((ang, mirror))
        return poses

# ---------- Bitmap helpers ----------
def _empty_mask(w:int, h:int):
    return [bytearray(w) for _ in range(h)]

def rasterize_polygon_to_mask(mask, w, h, pts_scaled):
    if not pts_scaled: return
    ys = [p[1] for p in pts_scaled]
    y0 = max(0, int(math.floor(min(ys))))
    y1 = min(h-1, int(math.ceil(max(ys))))
    n = len(pts_scaled)
    for y in range(y0, y1+1):
        yscan = y + 0.5
        xs = []
        for i in range(n):
            x1,y1 = pts_scaled[i]
            x2,y2 = pts_scaled[(i+1)%n]
            if y1==y2: continue
            if y1>y2: x1,y1,x2,y2 = x2,y2,x1,y1
            if y1 <= yscan and yscan < y2:
                t = (yscan - y1) / (y2 - y1)
                xs.append(x1 + t*(x2 - x1))
        if not xs: continue
        xs.sort()
        for i in range(0, len(xs), 2):
            x_start = int(math.floor(xs[i]))
            x_end   = int(math.ceil(xs[i+1])) - 1 if i+1 < len(xs) else x_start
            if x_end < 0 or x_start >= w: continue
            x_start = max(0, x_start); x_end = min(w-1, x_end)
            row = mask[y]
            for x in range(x_start, x_end+1):
                row[x] = 1

def rasterize_loops(loops: List[Loop], scale: float):
    allpts = [p for lp in loops for p in lp]
    if not allpts: return _empty_mask(1,1),1,1
    minx,miny,maxx,maxy = bbox_of_points(allpts)
    loops0 = [[(x-minx, y-miny) for (x,y) in lp] for lp in loops]
    pw = max(1, int(math.ceil((maxx-minx)*scale)))
    ph = max(1, int(math.ceil((maxy-miny)*scale)))
    mask = _empty_mask(pw, ph)
    if loops0:
        outer = loops0[0]
        outer_px = [(x*scale, y*scale) for x,y in outer]
        rasterize_polygon_to_mask(mask, pw, ph, outer_px)
        for hole in loops0[1:]:
            hole_px = [(x*scale, y*scale) for x,y in hole]
            hmask = _empty_mask(pw, ph)
            rasterize_polygon_to_mask(hmask, pw, ph, hole_px)
            for y in range(ph):
                row = mask[y]; hr = hmask[y]
                for x in range(pw):
                    if hr[x]: row[x] = 0
    return mask, pw, ph

def rasterize_outer_only(loops: List[Loop], scale: float):
    """Outer loop filled, holes ignored (used to forbid hole nesting)."""
    if not loops: return _empty_mask(1,1), 1, 1
    outer = loops[0]
    minx,miny,maxx,maxy = bbox_of_points(outer)
    pw = max(1, int(math.ceil((maxx-minx)*scale)))
    ph = max(1, int(math.ceil((maxy-miny)*scale)))
    mask = _empty_mask(pw, ph)
    pts = [((x-minx)*scale, (y-miny)*scale) for (x,y) in outer]
    rasterize_polygon_to_mask(mask, pw, ph, pts)
    return mask, pw, ph

def dilate_mask(mask, w, h, r):
    if r <= 0: return mask
    out = _empty_mask(w,h)
    offsets=[]
    rr = r*r
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dx*dx + dy*dy <= rr:
                offsets.append((dx,dy))
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x]:
                for dx,dy in offsets:
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < w and 0 <= yy < h:
                        out[yy][xx] = 1
    return out

def any_overlap(occ, part_mask, ox, oy, pw, ph):
    for y in range(ph):
        row_p = part_mask[y]
        row_o = occ[oy+y]
        for x in range(pw):
            if row_p[x] and row_o[ox+x]:
                return True
    return False

def bl_place(occ, part_mask):
    H = len(occ); W = len(occ[0]) if H>0 else 0
    ph = len(part_mask); pw = len(part_mask[0]) if ph>0 else 0
    for y in range(0, H - ph + 1):
        for x in range(0, W - pw + 1):
            if not any_overlap(occ, part_mask, x, y, pw, ph):
                return x, y
    return None

def or_mask_inplace(occ, part_mask, ox, oy):
    ph = len(part_mask); pw = len(part_mask[0]) if ph>0 else 0
    for y in range(ph):
        row_p = part_mask[y]
        row_o = occ[oy+y]
        for x in range(pw):
            if row_p[x]:
                row_o[ox+x] = 1

# small safety dilation cache (1 px safety to avoid aliasing leaks)
_disk_offsets_cache: Dict[int, List[Tuple[int,int]]] = {}
def _disk_offsets(r: int):
    if r in _disk_offsets_cache: return _disk_offsets_cache[r]
    offs=[]
    rr=r*r
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dx*dx + dy*dy <= rr:
                offs.append((dx,dy))
    _disk_offsets_cache[r]=offs
    return offs

def or_dilated_mask_inplace(occ, raw_mask, ox, oy, r):
    if r<=0:
        or_mask_inplace(occ, raw_mask, ox, oy); return
    offs = _disk_offsets(r)
    ph = len(raw_mask); pw = len(raw_mask[0]) if ph>0 else 0
    H = len(occ); W = len(occ[0]) if H>0 else 0
    for y in range(ph):
        row_p = raw_mask[y]
        for x in range(pw):
            if row_p[x]:
                bx = ox + x; by = oy + y
                for dx,dy in offs:
                    xx = bx + dx; yy = by + dy
                    if 0 <= xx < W and 0 <= yy < H:
                        occ[yy][xx] = 1

# ---------- Packer: Bitmap core (exact spacing + 1px safety) ----------



def pack_bitmap_core(ordered_parts: List['Part'], W: float, H: float, spacing: float, scale: int,
                     progress=None, progress_total=None, progress_prefix="",
                     mask_ops: Optional['TorchMaskOps'] = None):
    Wpx = max(1, int(math.ceil(W * scale)))
    Hpx = max(1, int(math.ceil(H * scale)))
    r_px = int(math.ceil(spacing * scale))  # dilation radius for spacing
    SAFETY_PX = 1                           # tiny safety to kill aliasing leaks

    sheets_occ_raw: List[Any] = []  # stored raw occupancy
    sheets_occ_safe: List[Any] = []  # safety-extended occupancy (1px) for overlap tests
    sheets_out = []
    sheets_count = 0

    def ensure_sheet():
        nonlocal sheets_count
        if len(sheets_occ_raw) <= sheets_count:
            if mask_ops:
                sheets_occ_raw.append(mask_ops.zeros(Hpx, Wpx))
                sheets_occ_safe.append(mask_ops.zeros(Hpx, Wpx))
            else:
                sheets_occ_raw.append(_empty_mask(Wpx, Hpx))
                sheets_occ_safe.append(_empty_mask(Wpx, Hpx))
            sheets_out.append([])
        return sheets_occ_raw[sheets_count], sheets_occ_safe[sheets_count], sheets_out[sheets_count]

    placed_count = 0
    total_parts = progress_total if progress_total is not None else len(ordered_parts)






    for p in ordered_parts:
        placed = False
        for ang in p.candidate_angles():
            key = (scale, ang)



            if key not in p._cand_cache:
                w,h,loops = p.oriented(ang)

                raw, pw, ph = rasterize_loops(loops, scale)               # outer minus holes
                test = dilate_mask(raw, pw, ph, r_px)                      # spacing test
                if not ALLOW_NEST_IN_HOLES:
                    shell, _, _ = rasterize_outer_only(loops, scale)       # outer only
                else:
                    shell = raw
                p._cand_cache[key] = {'loops':loops,'raw':raw,'test':test,'shell':shell,'pw':pw,'ph':ph}
            cand = p._cand_cache[key]

            if mask_ops:
                if 'raw_tensor' not in cand:
                    cand['raw_tensor'] = mask_ops.mask_to_tensor(cand['raw'])
                if 'test_tensor' not in cand:
                    cand['test_tensor'] = mask_ops.mask_to_tensor(cand['test'])
                if 'shell_tensor' not in cand:
                    cand['shell_tensor'] = mask_ops.mask_to_tensor(cand['shell'])

            attempt_sheet = sheets_count
            while True:
                occ_raw, occ_safe, outlist = ensure_sheet()
                # test for free spot against safety occupancy
                if mask_ops:
                    pos = mask_ops.find_first_fit(occ_safe, cand['test_tensor'])
                else:
                    pos = bl_place(occ_safe, cand['test'])
                if pos is not None:
                    xpx, ypx = pos
                    # commit: raw into raw-occ; safety-dilated shell/raw into safety-occ
                    if mask_ops:
                        mask_ops.or_mask(occ_raw, cand['raw_tensor'], xpx, ypx)
                        mask_ops.or_dilated(occ_safe, cand['shell_tensor'], xpx, ypx, SAFETY_PX)
                    else:
                        or_mask_inplace(occ_raw,  cand['raw'],   xpx, ypx)
                        or_dilated_mask_inplace(occ_safe, cand['shell'], xpx, ypx, SAFETY_PX)
                    # record geometry in units
                    x_units = xpx / scale
                    y_units = ypx / scale
                    loops_t = [[(x + x_units, y + y_units) for (x,y) in lp] for lp in cand['loops']]
                    outlist.append({'sheet': sheets_count, 'loops': loops_t})


                    placed = True
                    placed_count += 1
                    if progress:
                        progress(f"{progress_prefix}Placing parts…\n"
                                 f"Placed: {placed_count}/{total_parts}\n"
                                 f"Current sheet: {sheets_count+1}\n"
                                 f"Part: {os.path.basename(p.name)}")
                    break
                else:
                    sheets_count += 1
                    if progress:
                        progress(f"{progress_prefix}Opening new sheet… now {sheets_count+1}\n"
                                 f"Placed: {placed_count}/{total_parts}")
                    if sheets_count > attempt_sheet + 25:
                        break
            if placed: break




        if not placed:
            # last resort: drop at (0,0) of a fresh sheet
            sheets_count += 1
            occ_raw, occ_safe, outlist = ensure_sheet()
            w,h,loops = p.oriented(0.0, False)
            raw, pw, ph = rasterize_loops(loops, scale)
            shell = rasterize_outer_only(loops, scale)[0] if not ALLOW_NEST_IN_HOLES else raw
            if mask_ops:
                raw_tensor = mask_ops.mask_to_tensor(raw)
                shell_tensor = mask_ops.mask_to_tensor(shell)
                mask_ops.or_mask(occ_raw, raw_tensor, 0, 0)
                mask_ops.or_dilated(occ_safe, shell_tensor, 0, 0, SAFETY_PX)
            else:
                or_mask_inplace(occ_raw, raw, 0, 0)
                or_dilated_mask_inplace(occ_safe, shell, 0, 0, SAFETY_PX)
            outlist.append({'sheet': sheets_count, 'loops': loops})

            placed_count += 1
            if progress:
                progress(f"{progress_prefix}Forced place on new sheet {sheets_count+1}\n"
                         f"Placed: {placed_count}/{total_parts}")

    used_sheets = sheets_count + 1 if sheets_out and sheets_out[0] else len(sheets_out)
    # utilization metric (raw pixels)
    if mask_ops:
        fill_pixels = sum(mask_ops.count_true(occ) for occ in sheets_occ_raw)
    else:
        fill_pixels = 0
        for occ in sheets_occ_raw:
            for row in occ:
                fill_pixels += sum(1 for v in row if v)


    placements = [{'sheet': i, 'loops': pl['loops']} for i, out in enumerate(sheets_out) for pl in out]
    return placements, used_sheets, fill_pixels




# ---------- Bitmap order optimization helpers ----------
def _seq_key(order: List['Part']):
    return tuple(p.uid for p in order)


def _result_is_better(candidate, incumbent):
    if candidate is None:
        return False
    if incumbent is None:
        return True
    _, cand_sheets, cand_fill = candidate
    _, inc_sheets, inc_fill = incumbent
    if cand_sheets != inc_sheets:
        return cand_sheets < inc_sheets
    return cand_fill > inc_fill


def _mutate_order(order: List['Part'], rnd: Random) -> List['Part']:
    n = len(order)
    if n <= 1:
        return list(order)
    op = rnd.random()
    if n == 2:
        op = 0.0
    if op < 0.4:
        i, j = rnd.sample(range(n), 2)
        new_order = list(order)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        return new_order
    elif op < 0.75:
        i, j = rnd.sample(range(n), 2)
        new_order = list(order)
        part = new_order.pop(i)
        new_order.insert(j, part)
        return new_order
    else:
        i, j = sorted(rnd.sample(range(n), 2))
        if i == j:
            return list(order)
        new_order = list(order)
        new_order[i:j+1] = reversed(new_order[i:j+1])
        return new_order


def _anneal_order(initial_order: List['Part'], evaluate_fn, rnd: Random, sheet_penalty: int,
                  progress=None, label="", max_iters: Optional[int] = None):
    order = list(initial_order)
    best_order = list(order)
    best_result = evaluate_fn(best_order, allow_progress=False)
    current_order = list(order)
    current_result = best_result

    n = len(order)
    if n <= 1:
        return best_order, best_result

    default_iters = max(8, min(24, n + 4))
    if max_iters is not None:
        base_iters = max(5, min(default_iters, max_iters))
    else:
        base_iters = default_iters
    temperature = max(1.0, n * 0.4)
    cooling = 0.9
    stall_limit = None

    def score(res):
        if res is None:
            return float('inf')
        _, sheets, fill = res
        return sheets * sheet_penalty - fill

    stall = 0
    for it in range(1, base_iters + 1):
        candidate_order = _mutate_order(current_order, rnd)
        candidate_result = evaluate_fn(candidate_order, allow_progress=False)

        if _result_is_better(candidate_result, current_result):
            current_order, current_result = candidate_order, candidate_result
        else:
            delta = score(candidate_result) - score(current_result)
            if delta < 0:
                accept_prob = 1.0
            else:
                if temperature <= 0:
                    accept_prob = 0.0
                else:
                    accept_prob = math.exp(-delta / temperature)
            if accept_prob > rnd.random():
                current_order, current_result = candidate_order, candidate_result

        if _result_is_better(current_result, best_result):
            best_order, best_result = list(current_order), current_result
            if progress:
                progress(f"{label}Anneal improvement: sheets={best_result[1]}, fill={best_result[2]}")
            stall = 0
        else:
            stall += 1

        if progress and it % max(6, base_iters // 3) == 0:
            progress(f"{label}Anneal {it}/{base_iters}: best sheets={best_result[1]}, fill={best_result[2]}")

        temperature *= cooling
        if temperature < 1e-4:
            temperature = 1e-4
        if stall_limit is None:
            stall_limit = max(3, base_iters // 2)
        if stall >= stall_limit:
            break

    return best_order, best_result

# ---------- Bitmap multi-try ----------



def pack_bitmap_multi(parts: List['Part'], W: float, H: float, spacing: float, scale: int,
                      tries: int, seed: Optional[int], progress=None,
                      mask_ops: Optional['TorchMaskOps'] = None):



    base = [p for p in parts if p.outer is not None]
    base.sort(key=lambda p: abs(polygon_area(p.outer)), reverse=True)
    rnd = Random(seed) if seed is not None else Random()
    total_parts = len(base)

    if total_parts == 0:
        return [], 0

    search_scale = scale
    if scale > 6:
        search_scale = max(6, scale // 2)

    Wpx = max(1, int(math.ceil(W * scale)))
    Hpx = max(1, int(math.ceil(H * scale)))
    sheet_penalty = Wpx * Hpx * 1000

    cache: Dict[Tuple[tuple, int], Tuple[List[dict], int, int]] = {}

    def evaluate(order: List['Part'], allow_progress: bool, prefix: str = "", use_scale: int = search_scale):
        key = (_seq_key(order), use_scale)
        if key in cache:
            return cache[key]
        if allow_progress and progress:



            result = pack_bitmap_core(order, W, H, spacing, use_scale,
                                      progress=progress,
                                      progress_total=total_parts,
                                      progress_prefix=prefix,
                                      mask_ops=mask_ops)
        else:
            result = pack_bitmap_core(order, W, H, spacing, use_scale, progress=None, mask_ops=mask_ops)



        cache[key] = result
        return result

    best_result = None
    best_order: Optional[List['Part']] = None

    heuristic_orders: List[Tuple[str, List['Part']]] = []
    heuristic_orders.append(("Area-desc ", list(base)))
    heuristic_orders.append(("Aspect-desc ", sorted(base, key=lambda p: max(p.w, p.h, p.obb_w, p.obb_h), reverse=True)))
    heuristic_orders.append(("Tall-first ", sorted(base, key=lambda p: p.h, reverse=True)))

    tries = max(1, tries)

    start_orders: List[Tuple[str, List['Part']]] = []
    for ho in heuristic_orders:
        if len(start_orders) >= tries:
            break
        start_orders.append(ho)
    while len(start_orders) < tries:
        idx = len(start_orders) - len(heuristic_orders) + 1
        start_orders.append((f"Random {max(1, idx)} ", rnd.sample(base, len(base))))

    attempts = max(1, len(start_orders))
    anneal_limit = max(4, min(8, total_parts + max(1, tries // 2)))
    last_start_result = None
    for t, (label, start_order) in enumerate(start_orders):
        if progress:
            progress(f"{label}placement trial {t+1}/{attempts}…")

        start_result = evaluate(start_order, allow_progress=False, prefix=f"{label}Try {t+1}/{attempts}\n", use_scale=search_scale)
        last_start_result = start_result
        if anneal_limit <= 0:
            order_after, result_after = start_order, start_result
        else:
            if t == 0:
                limit = anneal_limit
            elif t < len(heuristic_orders):
                limit = min(3, anneal_limit)
            else:
                limit = min(4, anneal_limit)
            if limit <= 1:
                order_after, result_after = start_order, start_result
            else:
                order_after, result_after = _anneal_order(
                    start_order,
                    lambda o, allow_progress=False: evaluate(o, allow_progress, prefix=label, use_scale=search_scale),
                    rnd,
                    sheet_penalty,
                    progress=progress,
                    label=label,
                    max_iters=limit
                )

        final_result = result_after if _result_is_better(result_after, start_result) else start_result
        final_order = order_after if final_result is result_after else start_order

        if _result_is_better(final_result, best_result):
            best_result = final_result
            best_order = final_order
            if progress:
                progress(f"{label}New global best: sheets={best_result[1]}, fill={best_result[2]}")
        elif progress and best_result:
            progress(f"{label}Result sheets={final_result[1]}, fill={final_result[2]} (best remains sheets={best_result[1]}, fill={best_result[2]})")

    if best_result is None:
        best_result = last_start_result
        best_order = start_orders[0][1] if start_orders else base

    final_order = best_order if best_order is not None else base
    final_result = evaluate(final_order, allow_progress=True, prefix="Final pass\n", use_scale=scale)
    return final_result[0], final_result[1]


# ---------- Shelf fallback ----------
def pack_shelves(parts: List['Part'], W: float, H: float, spacing: float):
    parts=sorted([p for p in parts if p.outer is not None],
                 key=lambda p: max(p.w,p.h,p.obb_w,p.obb_h),
                 reverse=True)
    placements=[]; sheet=0; shelf_y=0.0; shelf_h=0.0; cursor_x=0.0
    def new_sheet():
        nonlocal sheet,shelf_y,shelf_h,cursor_x
        sheet+=1; shelf_y=0.0; shelf_h=0.0; cursor_x=0.0
    for p in parts:
        cands=[]

        for ang, mirror in p.candidate_poses():
            w,h,_=p.oriented(ang, mirror)
            cands.append((ang, mirror, w, h))
        placed=False
        for ang, mirror, w, h in cands:
            if cursor_x + w + spacing <= W and shelf_y + max(shelf_h, h + spacing) <= H:
                _,_,loops = p.oriented(ang, mirror)
                placements.append({'sheet':sheet,'loops':[[(x+cursor_x,y+shelf_y) for x,y in lp] for lp in loops]})
                cursor_x += w + spacing; shelf_h = max(shelf_h, h + spacing)
                placed=True; break
        if placed: continue
        shelf_y += shelf_h; cursor_x = 0.0; shelf_h = 0.0
        for ang, mirror, w, h in cands:
            if w + spacing <= W and shelf_y + h + spacing <= H:
                _,_,loops = p.oriented(ang, mirror)
                placements.append({'sheet':sheet,'loops':[[(x+0.0,y+shelf_y) for x,y in lp] for lp in loops]})
                cursor_x = w + spacing; shelf_h = h + spacing
                placed=True; break
        if placed: continue
        new_sheet()
        ok=False
        for ang, mirror, w, h in cands:
            if w + spacing <= W and h + spacing <= H:
                _,_,loops = p.oriented(ang, mirror)
                placements.append({'sheet':sheet,'loops':[[(x+0.0,y+0.0) for x,y in lp] for lp in loops]})
                cursor_x = w + spacing; shelf_h = h + spacing
                ok=True; break
        if not ok:
            _,_,loops = p.oriented(0.0, False)
            placements.append({'sheet':sheet,'loops':[[(x,y) for x,y in lp] for lp in loops]})
            cursor_x = p.w + spacing; shelf_h = p.h + spacing

    sheets_used=(max((pl['sheet'] for pl in placements), default=-1))+1
    return placements, sheets_used

# ---------- DXF writer ----------
def write_r12_dxf(path, sheets, W, H, placements, margin):
    def w(f,c,v): f.write(f"{c}\n{v}\n")
    with open(path,'w',encoding='utf-8') as f:
        w(f,0,"SECTION"); w(f,2,"HEADER"); w(f,9,"$INSUNITS"); w(f,70,INSUNITS); w(f,0,"ENDSEC")
        w(f,0,"SECTION"); w(f,2,"TABLES"); w(f,0,"ENDSEC")
        w(f,0,"SECTION"); w(f,2,"ENTITIES")
        for s in range(sheets):
            sheet_ox = s*(W + 2*margin + SHEET_GAP)
            w(f,0,"POLYLINE"); w(f,8,"SHEET"); w(f,66,1); w(f,70,1)
            for x,y in [
                (sheet_ox,                 0),
                (sheet_ox + W + 2*margin, 0),
                (sheet_ox + W + 2*margin, H + 2*margin),
                (sheet_ox,                 H + 2*margin),
                (sheet_ox,                 0)
            ]:
                w(f,0,"VERTEX"); w(f,8,"SHEET"); w(f,10,x); w(f,20,y)
            w(f,0,"SEQEND")
        for pl in placements:
            ox = pl['sheet']*(W + 2*margin + SHEET_GAP) + margin
            oy = margin
            for lp in pl['loops']:
                w(f,0,"POLYLINE"); w(f,8,"NEST"); w(f,66,1); w(f,70,1)
                for x,y in ((x+ox, y+oy) for x,y in lp):
                    w(f,0,"VERTEX"); w(f,8,"NEST"); w(f,10,x); w(f,20,y)
                w(f,0,"SEQEND")
        w(f,0,"ENDSEC"); w(f,0,"EOF")

# ---------- main ----------
def main():

    prog = WinProgress("Nesting DXF… Please wait", 480, 220)
    prog.create()

    if not os.path.isdir(FOLDER):
        log(f"[ERROR] Folder not found: {FOLDER}")
        prog.update("Folder not found.\nCheck FOLDER path in the script.")
        _write_report(FOLDER, ["Status: failed (folder not found)"])
        prog.close(); return

    dxf_files = sorted([f for f in os.listdir(FOLDER)
                        if f.lower().endswith(".dxf") and f.lower() != "nested.dxf"])
    if not dxf_files:
        log(f"[WARN] No .dxf files found in: {FOLDER}")
        prog.update("No .dxf files found.\nAdd DXFs to the folder and rerun.")
        _write_report(FOLDER, ["Status: aborted (no DXFs found)"])
        prog.close(); return

    W_eff = SHEET_W - 2*SHEET_MARGIN
    H_eff = SHEET_H - 2*SHEET_MARGIN
    if W_eff <= 0 or H_eff <= 0:
        msg = f"[ERROR] SHEET_MARGIN={SHEET_MARGIN} leaves no usable area on a {SHEET_W}×{SHEET_H} sheet."
        log(msg)
        prog.update(msg)
        _write_report(FOLDER, ["Status: failed (invalid sheet margin)"])
        prog.close(); return


    parts: List[Part] = []
    skipped = 0

    prog.update(f"Reading DXFs… 0/{len(dxf_files)}")
    for idx, fn in enumerate(dxf_files, 1):
        prog.update(f"Reading DXFs… {idx}/{len(dxf_files)}\n{fn}")
        path = os.path.join(FOLDER, fn)
        loops, segs = parse_entities(path)
        if not loops and segs:
            loops = join_segments_to_loops(segs, JOIN_TOL)
        fallback_bbox=None
        if not loops and segs and FALLBACK_OPEN_AS_BBOX:
            pts = [pt for a,b in segs for pt in (a,b)]
            if pts:
                minx,miny,maxx,maxy = bbox_of_points(pts)
                if maxx>minx and maxy>miny:
                    fallback_bbox=(minx,miny,maxx,maxy)
        if not loops and fallback_bbox is None:
            log(f"[WARN] {fn}: no closed loops/usable geometry; skipped.")
            skipped += 1
            continue
        p = Part(fn, loops, fallback_bbox)
        if p.outer is None or p.w<=0 or p.h<=0:
            log(f"[WARN] {fn}: zero-sized after parsing; skipped.")
            skipped += 1; continue
        qty = read_qty_for_dxf(FOLDER, fn)
        for _ in range(qty):
            parts.append(p)




    if not parts:
        log("[WARN] Nothing to nest (no usable profiles).")
        prog.update("Nothing to nest.\nNo usable closed profiles were found.")
        prog.close(); return

    mask_ops: Optional['TorchMaskOps'] = None
    accel_note = "Acceleration: CPU bitmap evaluator"
    using_cuda = False
    device_pref = BITMAP_DEVICE.strip() if BITMAP_DEVICE else None
    if NEST_MODE.lower() == "bitmap":
        mask_ops = build_mask_ops(device_pref)
        if mask_ops:
            device_desc = f"{mask_ops.device}"
            if getattr(mask_ops.device, "type", "") == "cuda":
                using_cuda = True
                accel_note = f"Acceleration: CUDA GPU ({device_desc}) via PyTorch"
            else:
                accel_note = f"Acceleration: PyTorch device {device_desc}"
            log(f"[INFO] Bitmap accelerator active on {device_desc}.")
        else:
            if device_pref:
                log(f"[WARN] Requested Torch device '{device_pref}' is unavailable; using CPU bitmaps.")
            elif cuda_available():
                log("[WARN] CUDA runtime detected but PyTorch is unavailable; using CPU bitmaps.")

    prog.update(f"Starting nesting… {len(parts)} parts\nMode: {NEST_MODE}, Tries: {SHUFFLE_TRIES}")

    if NEST_MODE.lower()=="bitmap":
        if SHUFFLE_TRIES>1:
            placements, sheets = pack_bitmap_multi(parts, W_eff, H_eff, SPACING, PIXELS_PER_UNIT,
                                                   SHUFFLE_TRIES, SHUFFLE_SEED,
                                                   progress=prog.update,
                                                   mask_ops=mask_ops)
        else:
            placements, sheets = pack_bitmap_core(parts, W_eff, H_eff, SPACING, PIXELS_PER_UNIT,
                                                  progress=prog.update,
                                                  progress_total=len(parts),
                                                  mask_ops=mask_ops)[:2]
    else:
        placements, sheets = pack_shelves(parts, W_eff, H_eff, SPACING)


    if sheets <= 0:
        log("[WARN] Parts exist, but none fit on the sheet.")
        prog.update("Parts exist, but none fit on the sheet.")
        prog.close(); return

    out = os.path.join(FOLDER, "nested.dxf")
    prog.update(f"Writing output…\n{out}")
    write_r12_dxf(out, sheets, W_eff, H_eff, placements, SHEET_MARGIN)

    # Write report
    report_path = os.path.join(FOLDER, "nest_report.txt")
    _report_lines.insert(0, "=== Nesting Report ===")
    _report_lines.append("")
    _report_lines.append(f"Saved: {out}")
    _report_lines.append(f"Mode: {NEST_MODE}")
    _report_lines.append(f"Sheets: {sheets}")
    _report_lines.append(f"Margin: {SHEET_MARGIN}")

    _report_lines.append(f"Spacing: {SPACING}")
    _report_lines.append(f"Resolution: {PIXELS_PER_UNIT} px/unit")

    _report_lines.append(f"Shuffle tries: {SHUFFLE_TRIES}{'' if SHUFFLE_SEED is None else f' (seed {SHUFFLE_SEED})'}")
    _report_lines.append(f"Skipped DXFs: {skipped}")
    _report_lines.append(f"Rect-align mode: {RECT_ALIGN_MODE}")
    _report_lines.append(f"Allow mirror: {ALLOW_MIRROR}")
    _report_lines.append(f"Allow nest in holes: {ALLOW_NEST_IN_HOLES}")
    _report_lines.append(accel_note)
    if using_cuda:
        _report_lines.append("GPU acceleration engaged: NVIDIA CUDA device utilized for bitmap placement.")
    _report_lines.append("")


    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write("\n".join(_report_lines))
    except Exception as e:
        print(f"[WARN] Could not write report: {e}")

    prog.update("Done! Opening folder and report…")
    prog.close()

    try: os.startfile(FOLDER)
    except: pass
    try: os.startfile(report_path)
    except: pass



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Nest DXF parts onto rectangular sheets using bitmap search."
    )
    parser.add_argument(
        "--folder",
        default=FOLDER,
        help="Directory containing DXF files and optional quantity TXT files.",
    )
    parser.add_argument(
        "--sheet",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        type=float,
        default=(SHEET_W, SHEET_H),
        help="Sheet size in drawing units (width height).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=SHEET_MARGIN,
        help="Border margin to keep empty around each sheet.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=SPACING,
        help="Minimum spacing to keep between parts.",
    )
    parser.add_argument(
        "--nest-mode",
        choices=["bitmap", "shelf"],
        default=NEST_MODE,
        help="Select the nesting strategy.",
    )
    parser.add_argument(
        "--pixels-per-unit",
        type=int,
        default=PIXELS_PER_UNIT,
        help="Bitmap resolution used for placement evaluation.",
    )
    parser.add_argument(
        "--tries",
        type=int,
        default=SHUFFLE_TRIES,
        help="Number of shuffle tries to explore different part orders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SHUFFLE_SEED,
        help="Optional RNG seed for repeatable shuffles.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=BITMAP_EVAL_WORKERS,
        help="Number of worker processes used for bitmap evaluation (bitmap mode).",
    )
    parser.add_argument(
        "--device",
        default=BITMAP_DEVICE,
        help="Optional PyTorch device string for bitmap acceleration (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--device",
        default=BITMAP_DEVICE,
        help="Optional PyTorch device string for bitmap acceleration (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--device",
        default=BITMAP_DEVICE,
        help="Optional PyTorch device string for bitmap acceleration (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--device",
        default=BITMAP_DEVICE,
        help="Optional PyTorch device string for bitmap acceleration (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--allow-mirror",
        dest="allow_mirror",
        action="store_true",
        default=ALLOW_MIRROR,
        help="Enable mirrored placements when searching for fits.",
    )
    parser.add_argument(
        "--no-mirror",
        dest="allow_mirror",
        action="store_false",
        help="Disable mirrored placements.",
    )
    parser.add_argument(
        "--allow-hole-nesting",
        dest="allow_holes",
        action="store_true",
        default=ALLOW_NEST_IN_HOLES,
        help="Allow parts to be placed inside other parts' holes.",
    )
    parser.add_argument(
        "--forbid-hole-nesting",
        dest="allow_holes",
        action="store_false",
        help="Disallow parts being nested within holes.",
    )
    parser.add_argument(
        "--rect-align",
        choices=["off", "prefer", "force"],
        default=RECT_ALIGN_MODE,
        help="Control rectangle alignment heuristics.",
    )



    args = parser.parse_args()

    FOLDER = os.path.abspath(args.folder)
    SHEET_W, SHEET_H = map(float, args.sheet)
    SHEET_MARGIN = float(args.margin)
    SPACING = float(args.spacing)
    NEST_MODE = args.nest_mode
    PIXELS_PER_UNIT = max(1, int(args.pixels_per_unit))
    SHUFFLE_TRIES = max(1, int(args.tries))
    SHUFFLE_SEED = args.seed

    BITMAP_EVAL_WORKERS = args.workers
    BITMAP_DEVICE = args.device

    ALLOW_MIRROR = args.allow_mirror
    ALLOW_NEST_IN_HOLES = args.allow_holes
    RECT_ALIGN_MODE = args.rect_align

    main()


