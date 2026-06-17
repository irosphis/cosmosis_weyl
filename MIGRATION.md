# 科研数据迁移说明

> 原机器：`shh` — Ubuntu 20.04, x86_64, Linux 5.15
> 迁移时间：2026-06-17

---

## 一、~/work/ 目录总览

```
~/work/ (21G)
├── cosmosis_weyl/        9.8G  ⭐ 主项目
├── aips/                 4.1G  AIPS 射电数据处理
├── cc_desi/              2.4G  DESI 相关
├── cosmic-dipole-tension/ 1.9G  宇宙偶极子 tension
├── cobaya/               1.3G  Cobaya 宇宙学拟合
├── CDDR_Test/            1.2G  CDDR 测试
├── void-gravity/         349M  空洞引力
├── HDE-in-Modified-Gravity/ 140M  HDE 修正引力
├── master_thesis_zxw/     46M  ⭐ 硕士论文
├── DISCO-DJ/              34M  DISCO-DJ 模拟
├── frp_0.54.0_linux_amd64/ 31M  内网穿透工具（可删）
└── frp_0.54.0_linux_amd64.tar.gz  12M  （可删）
```

---

## 二、GitHub 已备份的项目

这些代码已经 push 到 GitHub，随时可以 clone 回来：

| 项目 | GitHub |
|------|--------|
| cosmosis_weyl | https://github.com/irosphis/cosmosis_weyl |
| CDDR_Test | https://github.com/irosphis/CDDR_Test |
| HDE-in-Modified-Gravity | https://github.com/irosphis/HDE-in-Modified-Gravity |
| master_thesis_zxw | https://github.com/irosphis/master-thesis |
| DISCO-DJ | https://github.com/cosmo-sims/DISCO-DJ |

---

## 三、需要物理备份的数据（~5G）

这些文件太大或没有 git，需要拷贝到移动硬盘/云盘：

### cosmosis_weyl/ 内部 (已从 git 排除)

| 路径 | 大小 | 说明 |
|------|------|------|
| chains/ | 72M | MCMC 采样链 |
| MyTest/chains/ | 793M | Nautilus/Polychord 链文件 |
| MyTest/*.fits | - | KiDS-1000 巡天数据 |
| output/ | 143M | 分析输出 + hdf5 |

### 外部仓库（可重新 clone，但如有本地修改需备份）

| 路径 | 大小 | 原始来源 |
|------|------|---------|
| kcap/ | 535M | `https://github.com/KiDS-WL/kcap.git` |
| cosmosis-standard-library/ | 639M | `https://github.com/cosmosis-developers/cosmosis-standard-library` |
| kcap_boss_module/ | 9.4M | `https://github.com/KiDS-WL/kcap_boss_module.git` |

### 无 git 的项目

| 路径 | 大小 | 内容 |
|------|------|------|
| cosmic-dipole-tension/ | 1.9G | 代码(2M) + .conda(920M) + data(940M) |
| aips/ | 4.1G | AIPS 软件 + DATA + FITS |
| cc_desi/ | 2.4G | DESI fastspec 数据 |
| cobaya/ | 1.3G | Cobaya + .venv + 宇宙学数据 |
| void-gravity/ | 349M | 含 python_libs |

> **精简建议**：以上项目中的 `.conda`、`.venv`、`python_libs` 都可以删掉，导出环境文件重建即可，能省下 ~2G 空间。

---

## 四、恢复指南

### 1. 克隆代码

```bash
cd ~/work
git clone https://github.com/irosphis/cosmosis_weyl.git
git clone https://github.com/irosphis/CDDR_Test.git
git clone https://github.com/irosphis/HDE-in-Modified-Gravity.git
git clone https://github.com/irosphis/master-thesis.git
```

### 2. 克隆依赖仓库

```bash
cd ~/work/cosmosis_weyl
git clone https://github.com/KiDS-WL/kcap.git
git clone https://github.com/cosmosis-developers/cosmosis-standard-library.git
git clone https://github.com/KiDS-WL/kcap_boss_module.git
```

### 3. 恢复 conda 环境

```bash
# cosmosis_weyl 的 conda 环境
conda env create -f ~/work/cosmosis_weyl/environment.yml
# 或直接在 cosmosis_weyl 目录下
mkdir -p conda-env
# 将备份的 conda 环境目录复制回 conda-env/

# cosmic-dipole-tension 的 conda 环境
# 放在 cosmic-dipole-tension/.conda/
```

### 4. 恢复数据文件

将物理备份的数据复制回对应位置：
- MCMC chains → `cosmosis_weyl/chains/` 和 `cosmosis_weyl/MyTest/chains/`
- FITS 数据 → `cosmosis_weyl/MyTest/`
- output → `cosmosis_weyl/output/`
- 巡天数据 → `cosmic-dipole-tension/data/`
- DESI 数据 → `cc_desi/`
- AIPS 数据 → `aips/DATA/`, `aips/FITS/`

---

## 五、环境信息

- **OS**: Ubuntu 20.04.6 LTS (Focal Fossa)
- **Kernel**: 5.15.0-139-generic
- **Arch**: x86_64
- **Python**: 3.14.3 (cosmosis_weyl conda 环境)
- **Conda 环境路径**: `cosmosis_weyl/conda-env/`

### cosmosis 依赖链

```
cosmosis → cosmosis-standard-library → camb, class, halofit, ...
         → kcap → kcap_boss_module (BOSS 模块)
                → cosebis (COSEBIs)
```

---

## 六、迁移检查清单

- [ ] cosmosis_weyl 代码 push 到 GitHub ✅
- [ ] CDDR_Test push 到 GitHub ✅
- [ ] master_thesis_zxw push 到 GitHub
- [ ] HDE-in-Modified-Gravity push 到 GitHub
- [ ] DISCO-DJ push 到 GitHub
- [ ] 导出 cosmosis_weyl conda 环境 yml
- [ ] 导出 cosmic-dipole-tension conda 环境
- [ ] chains 数据物理备份（~900M）
- [ ] FITS 数据物理备份
- [ ] cosmic-dipole-tension 代码 + 数据备份
- [ ] cobaya 代码备份
- [ ] cc_desi 数据备份
- [ ] void-gravity 备份
- [ ] aips 数据备份
