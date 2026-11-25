# -*- coding: utf-8 -*-
"""Draw the §7.2-7.4 figures from the exported measurement tables using Pillow.

Layout rules kept deliberately strict so the figures look tidy:
  * a fixed right-hand legend column, so the legend never sits on the data;
  * log axes where the quantity spans decades, with ticks at powers of ten;
  * dataset labels staggered on the x axis when two N values are close;
  * data labels are skipped when they would collide with one already drawn;
  * no LaTeX markup leaks into the bitmap.
"""
import io
import math
import os
import re

from PIL import Image, ImageDraw, ImageFont

SRC = r'D:\code\pbsg-stage2\section72-74.txt'
OUT = r'D:\code\pbsg-stage2\figs'

W, H = 1280, 720
BG = (255, 255, 255)
FG = (38, 42, 48)
MUTED = (108, 114, 122)
GRID = (228, 232, 237)
AXIS = (130, 136, 143)
COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
          (148, 103, 189), (140, 86, 75)]

FONTS = [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
         r'C:\Windows\Fonts\simsun.ttc']


def font(size, bold=False):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def plain(s):
    return s.replace('$', '').replace('{', '').replace('}', '').replace('\\cdot', '·')


def parse(path):
    txt = io.open(path, encoding='utf-8').read()
    blocks = re.split(r'={70,}\nSECTION (\d\.\d)\n={70,}\n', txt)
    out, i = {}, 1
    while i + 1 < len(blocks):
        lines = [l for l in blocks[i + 1].split('\n') if l.startswith('|')]
        if lines:
            head = [c.strip() for c in lines[0].strip('|').split('|')]
            out[blocks[i].strip()] = [
                dict(zip(head, [c.strip() for c in l.strip('|').split('|')]))
                for l in lines[2:]]
        i += 2
    return out


class Panel:
    def __init__(self, title, sub):
        self.img = Image.new('RGB', (W, H), BG)
        self.d = ImageDraw.Draw(self.img)
        self.t = font(29)
        self.s = font(17)
        self._used = []
        self.d.text((W / 2, 32), plain(title), font=self.t, fill=FG, anchor='mm')
        if sub:
            self.d.text((W / 2, 64), plain(sub), font=self.s, fill=MUTED, anchor='mm')

    def box(self, left=152, top=104, right=300, bottom=132):
        return (left, top, W - right, H - bottom)

    def axes(self, box, xlab, ylab, xlo, xhi, ylo, yhi, logx=False, logy=False,
             yticks=None, ylabels=None):
        d, x0, y0, x1, y1 = self.d, *box
        f = font(17)
        d.rectangle(box, outline=AXIS, width=1)
        ticks = yticks if yticks else [ylo + (yhi - ylo) * i / 4 for i in range(5)]
        for k, t in enumerate(ticks):
            fy = self._fy(t, ylo, yhi, logy)
            y = y1 - fy * (y1 - y0)
            d.line([(x0, y), (x1, y)], fill=GRID)
            lab = ylabels[k] if ylabels else self._fmt(t)
            d.text((x0 - 12, y), lab, font=f, fill=FG, anchor='rm')
        d.line([(x0, y1), (x1, y1)], fill=AXIS)
        d.line([(x0, y0), (x0, y1)], fill=AXIS)
        if xlab:
            d.text(((x0 + x1) / 2, y1 + 82), plain(xlab), font=font(19), fill=FG,
                   anchor='mm')
        if ylab:
            d.text((14, (y0 + y1) / 2), plain(ylab), font=font(19), fill=FG, anchor='lm')
        self.xlo, self.xhi, self.ylo, self.yhi = xlo, xhi, ylo, yhi
        self.logx, self.logy = logx, logy
        return box

    @staticmethod
    def _fy(v, lo, hi, logy):
        if logy:
            return (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (v - lo) / (hi - lo)

    @staticmethod
    def _fmt(t):
        if t == 0:
            return '0'
        if t >= 1000:
            return '%d' % t
        if t >= 1:
            return ('%.1f' % t).rstrip('0').rstrip('.')
        return '%.2f' % t

    def xy(self, box, x, y):
        x0, y0, x1, y1 = box
        if self.logx:
            fx = (math.log10(x) - math.log10(self.xlo)) / (math.log10(self.xhi) - math.log10(self.xlo))
        else:
            fx = (x - self.xlo) / (self.xhi - self.xlo)
        fy = self._fy(y, self.ylo, self.yhi, self.logy)
        return x0 + fx * (x1 - x0), y1 - fy * (y1 - y0)

    def label(self, px, py, text, color, size=16, prefer='up'):
        """Draw a data label only if it does not overlap one already placed."""
        f = font(size)
        bb = self.d.textbbox((0, 0), text, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        for dy in ((-h - 10, 'up'), (10, 'down')):
            y = py + dy[0] if dy[1] == 'up' else py + 10
            cand = (px - w / 2, y - h / 2, px + w / 2, y + h / 2)
            if not any(self._hit(cand, u) for u in self._used):
                self._used.append(cand)
                self.d.text((px, y), text, font=f, fill=color, anchor='mm')
                return True
        return False

    @staticmethod
    def _hit(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def legend(self, items, box):
        x = box[2] + 26
        y = box[1] + 16
        f = font(17)
        for name, col in items:
            self.d.line([(x, y), (x + 26, y)], fill=col, width=4)
            self.d.ellipse([x + 9, y - 5, x + 19, y + 5], fill=col)
            for k, line in enumerate(name.split('\n')):
                self.d.text((x + 36, y + k * 20), plain(line), font=f, fill=FG,
                            anchor='lm')
            y += 30 + 20 * (name.count('\n'))

    def save(self, name):
        p = os.path.join(OUT, name)
        self.img.save(p)
        return p


# ------------------------------------------------------------------ fig 1
def fig_scored(secs):
    N = {'PEMS08': 170, 'PEMS04': 307, 'PEMS03': 358, 'PEMS07': 883}
    order = ['PEMS08', 'PEMS04', 'PEMS03', 'PEMS07']
    series = [('B Dense-Adaptive', 'Dense / Global-TopK\n(全部 N(N-1) 对)'),
              ('E PBSG-m4', 'PBSG-m=4'), ('D PBSG-m0', 'PBSG-m=0'),
              ('A Road-Fixed', 'Road-Fixed')]
    p = Panel('打分阶段实际评估的节点对数',
              'log-log；Dense 与 Global-TopK 都必须打完全部 N(N-1) 对，PBSG 只打候选集')
    box = p.box(right=310, bottom=112)
    p.axes(box, '节点数 N（log）', '打分节点对数（log）', 150, 1000, 400, 2e6,
           logx=True, logy=True,
           yticks=[400, 1000, 4000, 16000, 64000, 256000, 1e6],
           ylabels=['400', '1k', '4k', '16k', '64k', '256k', '1M'])
    items = []
    for i, (m, lab) in enumerate(series):
        pts = []
        for ds in order:
            for r in secs['7.2']:
                if r['数据集'] == ds and r['方法'] == m:
                    pts.append((N[ds], int(r['打分数'].replace(',', ''))))
        pts.sort()
        xy = [p.xy(box, x, y) for x, y in pts]
        p.d.line(xy, fill=COLORS[i], width=3)
        for (px, py), (x, y) in zip(xy, pts):
            p.d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=COLORS[i])
            p.label(px, py, '{:,}'.format(y), COLORS[i])
        items.append((lab, COLORS[i]))
    for ds in order:
        px, _ = p.xy(box, N[ds], p.ylo)
        p.d.line([(px, box[3]), (px, box[3] + 7)], fill=AXIS)
        p.d.text((px, box[3] + 20), '%s\nN=%d' % (ds, N[ds]), font=font(15), fill=MUTED,
                 anchor='ma')
    p.legend(items, box)
    return p.save('sec7_scored_pairs.png')


# ------------------------------------------------------------------ fig 2
def fig_build(secs):
    N = {'PEMS08': 170, 'PEMS04': 307, 'PEMS03': 358, 'PEMS07': 883}
    order = ['PEMS08', 'PEMS04', 'PEMS03', 'PEMS07']
    series = ['A Road-Fixed', 'B Dense-Adaptive', 'C Global-TopK', 'D PBSG-m0',
              'E PBSG-m4']
    p = Panel('线上构图时间（builder 单独计时）',
              'warm-up 10 次 + 正式 100 次，每次前后 cuda.synchronize；固定 embedding 与候选索引')
    box = p.box(right=310, bottom=112)
    p.axes(box, '节点数 N（log）', '构图 median（ms）', 150, 1000, 0.0, 0.9,
           logx=True, logy=False, yticks=[0, 0.2, 0.4, 0.6, 0.8])
    items = []
    for i, m in enumerate(series):
        pts = []
        for ds in order:
            for r in secs['7.3']:
                if r['数据集'] == ds and r['方法'] == m:
                    pts.append((N[ds], float(r['构图 median ms'])))
        pts.sort()
        xy = [p.xy(box, x, y) for x, y in pts]
        p.d.line(xy, fill=COLORS[i], width=3)
        for (px, py), (x, y) in zip(xy, pts):
            p.d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=COLORS[i])
            p.label(px, py, '%.3f' % y, COLORS[i], size=15)
        items.append((m, COLORS[i]))
    for ds in order:
        px, _ = p.xy(box, N[ds], 0)
        p.d.text((px, box[3] + 20), '%s\nN=%d' % (ds, N[ds]), font=font(15), fill=MUTED,
                 anchor='ma')
    p.legend(items, box)
    return p.save('sec7_build_time.png')


# ------------------------------------------------------------------ fig 3
def fig_storage(secs):
    N07 = 883
    series = [('B Dense-Adaptive', 'B Dense'), ('C Global-TopK', 'C Global-TopK'),
              ('D PBSG-m0', 'D PBSG-m0'), ('E PBSG-m4', 'E PBSG-m4')]
    vals = []
    for m, lab in series:
        for r in secs['7.2']:
            if r['数据集'] == 'PEMS07' and r['方法'] == m:
                vals.append((lab, float(r['实际邻接 KB'].replace(',', '')),
                             float(r['稀疏等价 KB'].replace(',', ''))))
    p = Panel('PEMS07（N=883）邻接存储：当前实现 vs 按 COO 存的等价大小',
              '每边 20 B（两个 int64 索引 + 一个 float32 权重）；纵轴 log')
    box = p.box(right=310, bottom=120)
    p.axes(box, '', 'KB（log）', 0, len(vals), 30, 40000, logy=True,
           yticks=[30, 100, 300, 1000, 3000, 10000, 30000],
           ylabels=['30', '100', '300', '1k', '3k', '10k', '30k'])
    gw = (box[2] - box[0]) / len(vals)
    bw = gw * 0.24
    for i, (lab, act, sp) in enumerate(vals):
        cx = box[0] + gw * (i + 0.5)
        for j, (v, col) in enumerate([(act, COLORS[1]), (sp, COLORS[0])]):
            px = cx - bw * 1.08 + j * bw * 1.12
            _, py = p.xy(box, 0, v)
            p.d.rectangle([px, py, px + bw, box[3]], fill=col)
            p.d.text((px + bw / 2, py - 6), '%.0f' % v, font=font(16), fill=col,
                     anchor='mb')
        p.d.text((cx, box[3] + 22), lab, font=font(17), fill=FG, anchor='ma')
    p.legend([('实际邻接（当前实现）', COLORS[1]), ('稀疏等价 COO', COLORS[0])], box)
    return p.save('sec7_storage.png')


# ------------------------------------------------------------------ fig 4
def fig_decomp(secs):
    order = ['PEMS08', 'PEMS04', 'PEMS03', 'PEMS07']
    stages = [('graph %', '构图'), ('fwd %', '前向'), ('bwd %', '反向'),
              ('opt %', '优化器'), ('data %', '数据加载')]
    cols = [COLORS[3], COLORS[0], COLORS[1], COLORS[2], COLORS[4]]
    recs = []
    for ds in order:
        for r in secs['7.4']:
            if r['数据集'] == ds and r['方法'] == 'E PBSG-m4':
                recs.append((ds, [float(r[k]) for k, _ in stages]))
    p = Panel('PBSG-m=4 每 epoch 五段耗时分解',
              '插桩训练（每段前后 synchronize），括号外为该段占比 %')
    box = p.box(right=310, bottom=120)
    p.axes(box, '', '占比 %', 0, len(recs), 0, 100, yticks=[0, 20, 40, 60, 80, 100])
    gw = (box[2] - box[0]) / len(recs)
    bw = gw * 0.44
    for i, (ds, pcts) in enumerate(recs):
        cx = box[0] + gw * (i + 0.5)
        y = box[3]
        for (k, lab), col, pct in zip(stages, cols, pcts):
            h = pct / 100.0 * (box[3] - box[1])
            p.d.rectangle([cx - bw / 2, y - h, cx + bw / 2, y], fill=col,
                          outline=BG, width=1)
            if pct >= 4.5:
                p.d.text((cx, y - h / 2), '%.1f' % pct, font=font(16),
                         fill=(255, 255, 255), anchor='mm')
            y -= h
        p.d.text((cx, box[3] + 22), ds, font=font(19), fill=FG, anchor='ma')
    p.legend([(lab, c) for (k, lab), c in zip(stages, cols)], box)
    return p.save('sec7_epoch_decomp.png')


def main():
    os.makedirs(OUT, exist_ok=True)
    secs = parse(SRC)
    for f in (fig_scored, fig_build, fig_storage, fig_decomp):
        print('wrote', f(secs))


if __name__ == '__main__':
    main()



