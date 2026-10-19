"""可视化预生成样本 (配色/坐标沿用 visualization/paper_figs/style.py)。
用法: python vis_samples.py SAMPLES.h5 OUT.png i,j,k   (i,j,k 为数据集下标，不是 h5 行号)
注意: 网格纬度是从北到南存储 (第 0 行 = 24°N)，所以 imshow 用 origin='upper' 且 extent 的 bottom/top = 15/24。"""
import sys
ROOT = '/data1/user/lz/SST_Data_Imputation/Data_Imputation'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/visualization/paper_figs')
import numpy as np, h5py, matplotlib; matplotlib.use('Agg')
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import CMAPS, LAND_COLOR, MISSING_OVERLAY, CAT_OBSERVED, CAT_KNN, CAT_MISSING, apply_paper_style, setup_geo_ax
from datasets.single_model_dataset import SingleModelDataset, MEAN, STD

apply_paper_style()
path, out, ids = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
ds = SingleModelDataset(path)
with h5py.File(ROOT + '/experiments/hourly_store_v1/sst_hourly.h5', 'r') as f:
    lat, lon = f['lat'][:], f['lon'][:]
assert lat[0] > lat[-1] and lon[0] < lon[-1]                       # 北→南、西→东
EXT = [lon.min(), lon.max(), lat.min(), lat.max()]                  # left, right, bottom, top


def cm(c):
    c = c.copy(); c.set_bad(LAND_COLOR); return c


SST, ERR, CNT = cm(CMAPS['sst']), cm(CMAPS['sst_balance']), cm(CMAPS['sst_thermal'])
GRAY = mpl.colors.ListedColormap([MISSING_OVERLAY]); GRAY.set_bad((0, 0, 0, 0))
CAT = mpl.colors.ListedColormap([CAT_OBSERVED, CAT_KNN, CAT_MISSING]); CAT.set_bad(LAND_COLOR)
OBS = mpl.colors.ListedColormap(['#1a1a1a', '#f0f0f0']); OBS.set_bad(LAND_COLOR)
FILLED = h5py.File(ROOT + '/experiments/filled_store_v1/filled_hourly.h5', 'r')
HOLE = mpl.colors.ListedColormap([CAT_KNN]); HOLE.set_bad((0, 0, 0, 0))
titles = ['(a) Original observation', '(b) After masking', '(c) Masked, then filled\n(model input)',
          '(d) Filled without masking\n(real-frame fill)', '(e) (c) $-$ truth, in hole', '(f) Base field B $-$ truth, in hole',
          '(g) Observation count C']
n, m = len(ids), len(titles)
fig, ax = plt.subplots(n, m, figsize=(2.9 * m, 3.4 * n), squeeze=False)
for r, i in enumerate(ids):
    b = ds[i]; ocean = b['land_mask'].numpy() == 0; slot = int(b['t'])
    tr = b['ground_truth_sst'].numpy() * STD + MEAN; inp = b['input_sst_seq'][-1].numpy() * STD + MEAN
    base = b['base_sst'].numpy() * STD + MEAN; hole = b['hole'].numpy() > 0; lm = b['loss_mask'].numpy() > 0
    has_truth = (b['loss_mask'].numpy() + b['original_obs_mask'].numpy()) > 0
    q = FILLED['sst_filled'][slot]; fvalid = FILLED['valid'][slot] > 0
    real_fill = np.where(q != -32768, q * 0.01 + 273.15, np.nan) if fvalid else np.full(tr.shape, np.nan)
    land = lambda a: np.ma.masked_where(~ocean, a)
    ones = lambda mask: np.ma.masked_where(~mask, np.ones(hole.shape))
    nat = ocean & ~has_truth                                        # 原始帧本来就无观测 (真实云)
    vmin, vmax = np.percentile(tr[lm], [1, 99]) if lm.any() else (290, 300)
    e1 = np.where(lm, inp - tr, np.nan); e2 = np.where(lm, base - tr, np.nan)
    panels = [(land(np.where(has_truth, tr, np.nan)), SST, vmin, vmax),
              (land(np.where(has_truth & ~hole, tr, np.nan)), SST, vmin, vmax),
              (land(inp), SST, vmin, vmax), (land(real_fill), SST, vmin, vmax),
              (land(e1), ERR, -1.5, 1.5), (land(e2), ERR, -1.5, 1.5),
              (land(b['c_seq'][0].numpy() * 48), CNT, 0, 48)]
    ims = []
    for c, (a, cmap, lo, hi) in enumerate(panels):
        im = ax[r, c].imshow(a, origin='upper', extent=EXT, cmap=cmap, vmin=lo, vmax=hi, aspect='auto'); ims.append(im)
        if c in (0, 1):
            ax[r, c].imshow(ones(nat), origin='upper', extent=EXT, cmap=GRAY, vmin=0, vmax=1, aspect='auto')   # 真实云: 灰
        if c == 1:
            ax[r, c].imshow(ones(hole), origin='upper', extent=EXT, cmap=HOLE, vmin=0, vmax=1, aspect='auto')  # 人工洞: 赤陶色
        if c == 3 and not fvalid:
            ax[r, c].text(0.5, 0.5, 'no real-frame fill\nfor this slot', transform=ax[r, c].transAxes, ha='center', fontsize=8)
        setup_geo_ax(ax[r, c], lon, lat, draw_xlabel=(r == n - 1), draw_ylabel=(c == 0), step=2)
        if r < n - 1: ax[r, c].set_xticklabels([])
        if c > 0: ax[r, c].set_yticklabels([])
        if r == 0: ax[r, c].set_title(titles[c], fontsize=9)
    for k, lab in ((3, 'SST (K), panels a-d'), (5, 'Error (K), panels e-f'), (6, 'Obs. count (48 h)')):
        cax = ax[r, k].inset_axes([1.04, 0.0, 0.05, 1.0]); cb = fig.colorbar(ims[k], cax=cax)
        cb.set_label(lab, fontsize=7); cb.ax.tick_params(labelsize=7)
    t = ds.times[slot]
    ax[r, 0].text(0.02, 0.03, f"{t:%Y-%m-%d %H:00} UTC\n{'large' if b['big'] > 0 else 'small'} hole, {hole.sum() / ocean.sum():.0%} of ocean\n"
                  f"RMSE (hole): filled {np.sqrt(np.nanmean(e1 ** 2)):.2f} K, B {np.sqrt(np.nanmean(e2 ** 2)):.2f} K",
                  transform=ax[r, 0].transAxes, fontsize=6.5, va='bottom', ha='left', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
fig.legend(handles=[Patch(fc=CAT_KNN, label='Artificial hole (masked)'), Patch(fc=MISSING_OVERLAY, label='Natural cloud (no observation)'),
                    Patch(fc=LAND_COLOR, label='Land')],
           loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(left=0.05, right=0.985, top=0.95, bottom=0.07, wspace=0.62, hspace=0.12)
plt.savefig(out, dpi=110); print(out)
