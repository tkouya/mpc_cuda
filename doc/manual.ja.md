# mpc_cuda マニュアル

**バージョン 0.0.1**

> これは `doc/manual.md`（英語版）の日本語訳です。内容に差異がある場合は英語版が正となります。

mpc_cuda は GNU 多倍長演算スタック — **mini-gmp**（任意精度整数）、**MPFR**（正しく丸められ
る実数浮動小数点）、**MPC**（複素数）— を *CUDA カーネルの内部で* 実行します。これにより、各
GPU スレッドがそれぞれ独自の任意精度計算を行えます。すべての演算は、テスト済みの入力につい
てホストのライブラリとビット単位で一致します。

デバイスコードは、再実行可能な変換スクリプト（`tools/cudafy_*.py`）によって、無改変の上流
ソースから生成されます。mpc_cuda は手作業のフォークではないため、将来の上流リリースに対して
再生成できます。

---

## 1. 目次

1. [必要要件](#2-必要要件)
2. [ビルドとインストール](#3-ビルドとインストール)
3. [`cu_` 名前空間と共存](#4-cu_-名前空間と共存)
4. [カーネルを書く](#5-カーネルを書く)
5. [バンプアリーナ：概念とサイズ設定](#6-バンプアリーナ概念とサイズ設定)
6. [プログラムのリンク](#7-プログラムのリンク)
7. [API の範囲](#8-api-の範囲)
8. [デモとベンチマーク](#9-デモとベンチマーク)
9. [動作原理](#10-動作原理)
10. [制限事項](#11-制限事項)
11. [ライセンス](#12-ライセンス)

---

## 2. 必要要件

* NVIDIA GPU と **CUDA Toolkit**（`nvcc`）。デフォルトのターゲットアーキテクチャは
  `sm_121`（NVIDIA GB10）です。別のものは `--with-cuda-arch=` で指定します。
* **Python 3**（変換スクリプトを実行します）。
* C++ コンパイラ（`g++`）と `nm`（binutils）。
* 同梱の上流ソース `gmp-6.3.0/mini-gmp`、`mpfr-4.2.2/src`、`mpc-1.4.1/src`
  （配布物に含まれています。`--with-*-src` で上書きできます）。
* 任意：**システムの** `libgmp`/`libmpfr`/`libmpc` とその開発ヘッダ
  （`<mpfr.h>`、`<mpc.h>`）— `make coexist` / `make cputest` / `make cpubench` の場合のみ
  必要です。これらは GPU 移植を CPU ライブラリと照合します。

---

## 3. ビルドとインストール

mpc_cuda は autoconf ベースのビルド（手書きの `Makefile.in`、automake は不使用。ビルドは
`nvcc` と変換スクリプトで駆動されるため）を採用しています。

```sh
./autogen.sh          # git チェックアウトからビルドする場合のみ（./configure を再生成）
./configure           # オプションは下記参照
make                  # libmpc_cuda.a + libmpc_cuda.so をビルド（再生成 + コンパイル）
make check            # テストスイートをビルドして実行
make demos            # AXPY / 行列 / ベンチマークのデモをビルド
sudo make install     # ヘッダ、libmpc_cuda.a/.so、mpc_cuda-link、mpc_cuda.pc をインストール
```

### configure オプション

| オプション | 意味 | デフォルト |
|--------|---------|---------|
| `--with-cuda-arch=ARCH` | ターゲット GPU 演算アーキテクチャ（`sm_121`、`sm_90`、…） | `sm_121` |
| `--with-gmp-src=DIR`  | mini-gmp ソースディレクトリ | `gmp-6.3.0/mini-gmp` |
| `--with-mpfr-src=DIR` | MPFR ソースディレクトリ     | `mpfr-4.2.2/src` |
| `--with-mpc-src=DIR`  | MPC ソースディレクトリ      | `mpc-1.4.1/src` |
| `--prefix=DIR`        | インストール先プレフィックス | `/usr/local` |
| `NVCC=…`、`PYTHON=…`  | 検出されたツールを上書き | 自動検出 |

> **同じツリーで複数のビルドを同時に実行しないでください**。いずれも共有の `build/`
> ディレクトリに対して再生成・コンパイルを行うため、競合します。`make` はデフォルトで
> 逐次実行なので、通常の `make -j1`（デフォルト）であれば問題ありません。

### `make` が生成するもの、`make install` がインストールするもの

`make` は、CUDA 用に変換した各ソースを **デバイス再配置可能（`-rdc`、`-fPIC`）オブジェクト**
にコンパイルし、それらを 2 つの形式の単一ライブラリにまとめます。

* **`libmpc_cuda.a`** — 静的アーカイブ。CUDA プログラムがリンクする形式です。
  `nvcc -rdc=true … libmpc_cuda.a` がこれをデバイスリンクします。`nvlink` は、あなたの
  カーネルが参照するデバイスオブジェクトだけをアーカイブから正確に取り出します。
* **`libmpc_cuda.so`** — 共有ライブラリ。一部の上流ソースはデバイスコードへ移植できません
  （FILE\* I/O、欠けている mini-gmp ヘルパ：`mpc_pow`、`mpcb_*` ボール、`out_str`、…）。
  それらの欠落シンボルに *推移的に依存する* オブジェクトはバンドルから刈り取られるため、
  アーカイブは **自己完結的**（参照について閉じている）であり、`nvlink` も `ld` も未解決の
  内部シンボルに遭遇しません。`.so` は内部でデバイスリンク済みなので、`nvlink` は外部カーネ
  ルのために `.so` からデバイスコードを取り出すことは **できません** — デバイスリンクされる
  実行ファイルは `libmpc_cuda.a` を使う必要があります。`.so` は **ホスト専用** の利用者向けに
  ホストから呼べる（`__host__ __device__`）`cu_*` API を公開します。

`make install` は次のように配置します。

```
$(includedir)/mpc_cuda/            cuda_minigmp.h, cu_compat.h
$(includedir)/mpc_cuda/mpfr/       MPFR ヘッダ (mpfr.h, mpfr-impl.h, …)
$(includedir)/mpc_cuda/mpc/        MPC ヘッダ  (mpc.h, mpc-impl.h, …)
$(libdir)/libmpc_cuda.a            静的ライブラリ（これをデバイスリンクする）
$(libdir)/libmpc_cuda.so           共有ライブラリ（ホストから呼べる cu_* API）
$(datadir)/mpc_cuda/               mpfr_cuda_defs.txt
$(bindir)/mpc_cuda-link            リンク補助（パスが埋め込み済み）
$(libdir)/pkgconfig/mpc_cuda.pc    pkg-config メタデータ
```

---

## 4. `cu_` 名前空間と共存

エクスポートされるすべてのシンボルは **`cu_` 名前空間** にリネームされます。

| 上流 | mpc_cuda |
|----------|----------|
| `mpz_*`、`mpn_*`、`gmp_*` | `cu_mpz_*`、`cu_mpn_*`、`cu_gmp_*` |
| `mpfr_*` | `cu_mpfr_*` |
| `mpc_*`  | `cu_mpc_*`  |

これにより、1 つのプログラムが mpc_cuda（GPU 用）と **本物のシステムの
`libgmp`/`libmpfr`/`libmpc`**（CPU 用）の **両方** を、「multiple definition（多重定義）」の
衝突なしにリンクできます。**型**（`mpfr_t`、`mp_limb_t`）、**丸め/列挙マクロ**
（`MPFR_RNDN`、`MPC_RNDNN`）、およびアリーナ補助（`mpc_cuda_*`）は *リネームされません*。

* **`.cu`** ファイルでは、`cu_mpfr_*` / `cu_mpc_*` / `cu_mpz_*` を直接呼ぶか、
  `#include "mpc_cuda/cu_compat.h"` を使って、慣れ親しんだ `mpfr_*` の名前のまま書けます
  （これは単に `cu_` 版へ `#define` しているだけです）。
* システムライブラリを使う **`.cpp`** ファイルでは、システムの `<mpfr.h>` をインクルードし、
  通常どおり `mpfr_*` を呼びます。
* mpc_cuda のヘッダとシステムのヘッダを *同一の* 翻訳単位に **両方インクルードしないでください**
  （型 `mpfr_t` が二重定義されます）。それぞれ別の `.cu` / `.cpp` ファイルに分け、オブジェクト
  をリンクしてください。

`make coexist` は、GPU 側で `cu_mpfr` を、CPU 側でシステムの `libmpfr` を使って π と log 2 を
計算し、両者が一致することを示す単一バイナリを実演します。

---

## 5. カーネルを書く

### 最小の例

```cuda
#include <cstdio>
typedef long int gmp_randstate_t[1];   // ヘッダが期待する shim
#include "mpfr.h"
#include "mpc_cuda/cuda_minigmp.h"
#include "mpc_cuda/cu_compat.h"        // 任意：以下で mpfr_* と書けるようにする

__global__ void k(double *out)
{
    mpfr_t x, r;
    mpfr_init2(x, 256);                 // 256 ビット精度
    mpfr_init2(r, 256);
    mpfr_set_d(x, 2.0, MPFR_RNDN);
    mpfr_log(r, x, MPFR_RNDN);          // r = ln 2、正しく丸められる
    out[0] = mpfr_get_d(r, MPFR_RNDN);
    mpfr_clear(x); mpfr_clear(r);
}
```

### デバイスのリソース上限

深い MPFR/MPC の呼び出し連鎖と limb ヒープには、デフォルトより大きい上限が必要です。起動
*前* にホスト側で設定してください。

```cuda
cudaDeviceSetLimit(cudaLimitStackSize,      192 * 1024);          // 128–256 KB
cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256ull * 1024 * 1024);
```

MPC の複素数乗算はかなり深く再帰します — MPC では **128 KB 以上** のスタックを使ってください。

### スレッドごとのバンプアリーナ（速度のため推奨）

デフォルトでは、各 `mpfr_init2`/演算がデバイスの `malloc`/`free` を呼び、そのアロケータの
オーバーヘッドが支配的になります。mpc_cuda は任意の **スレッドごとのバンプアリーナ** を提供
します。確保はポインタの加算（バンプ）、`free` は何もしない（no-op）で、1 回のリセットで次の
処理単位のためにすべてを回収します。これが GPU 対 CPU の大きな高速化をもたらします。

ホスト側でインストールし、カーネル内で処理単位ごとにリセットします。

```cuda
// ホスト：起動する各スレッドに 1 つの slab
size_t SLAB = 32 * 1024;                 // スレッドあたりバイト数。ワークロードに合わせる
size_t nthreads = blocks * threads;
char   *arena; size_t *top;
cudaMalloc(&arena, nthreads * SLAB);
cudaMalloc(&top,   nthreads * sizeof(size_t));
cudaMemset(top, 0, nthreads * sizeof(size_t));
mpc_cuda_arena_base = arena;             // managed なグローバル変数
mpc_cuda_arena_slab = SLAB;
mpc_cuda_arena_top  = top;

// カーネル：グリッドストライドで、反復ごとにアリーナをリセット
for (int i = tid; i < N; i += stride) {
    mpc_cuda_arena_reset();              // 前の処理単位のスクラッチを回収
    ... 要素 i に対する mpfr/mpc の処理 ...
}
```

`mpc_cuda_arena_base == NULL` のままにすると、デバイスの `malloc`/`free` にフォールバック
します（正しいが、遅いだけ）。`SLAB` は 1 つの処理単位のピーク生存メモリを保持できるサイズ
にしてください。処理単位が slab を超過した場合、アロケータは `malloc` にフォールバックします。

**起動形状が重要です。** 深い呼び出し連鎖は limb を（グローバルメモリの）アリーナ slab に
保持するため、そのレイテンシを隠すには多くのワープが常駐している必要があります。小さな問題に
要素ごとに 1 スレッドを割り当てるのではなく、大きな問題に対して数千スレッドの *固定* プールを
グリッドストライドさせる形を推奨します。

### リダクション/累積ループ：スタック上の値

内側ループで累積する場合（内積、級数）、アキュムレータは `mpc_cuda_arena_reset()` の呼び出し
をまたいで生存する必要があります。永続的なオペランドを **スタック** に置いて、アリーナが 1 演算
分のスクラッチだけを保持するようにします。

* **実数：** MPFR の `MPFR_DECL_INIT(name, PREC)` を使います — 仮数部がローカル配列である
  スタック確保された `mpfr_t`（ヒープなし）です。
* **複素数：** 2 つの `mpfr_custom_init_set` 成分からスタック上の `mpc_t` を構築します
  （`demos/matvec_mpc.cu` の `MPC_DECL_INIT` マクロを参照）。

```cuda
MPFR_DECL_INIT(acc, PREC);
MPFR_DECL_INIT(t, PREC);
mpfr_set_zero(acc, 1);
for (int j = 0; j < n; ++j) {
    mpc_cuda_arena_reset();              // acc と t は生存する（スタック上にあるため）
    mpfr_mul(t, a[j], x[j], MPFR_RNDN);
    mpfr_add(acc, acc, t, MPFR_RNDN);
}
```

---

## 6. バンプアリーナ：概念とサイズ設定

第 5 節ではアリーナを *どうインストールするか* を示しました。本節では、それが *何であるか*、
そして最も重要な点として *どのくらいの大きさにすべきか* を説明します。これは性能に最も影響
する唯一の調整ツマミであり、最も設定を誤りやすいものです。

### 6.1 アリーナが解決する問題

すべての多倍長演算はメモリを確保します。`mpfr_t` は仮数部（limb 群）用のバッファを必要とし、
ほぼすべてのルーチンはその上にさらに 1 つ以上の一時値を確保します — `cu_mpfr_div` や
`cu_mpc_mul` を 1 回呼ぶだけで、呼び出し連鎖を降りていく過程で *数十個* もの小さなバッファを
確保・解放しうるのです。CPU ではこれは安価です。GPU ではそうではありません。デバイスの
`malloc`/`free` は、デバイス上のすべてのスレッドが共有する単一のグローバルな直列化アロケータ
であり、数千のスレッドが同時にこれを叩くと支配的なコストになります — このライブラリでは、
GPU が CPU 同等にとどまるか、~40 倍の高速化になるかの分かれ目がここでした。

**バンプアリーナ** はそのコストを取り除きます。デバイスヒープへ行く代わりに、すべての確保が
あらかじめ予約された GPU グローバルメモリのブロックから供給されます。

* **確保** = 要求サイズをスレッドごとのオフセットに加算（「バンプ」）し、加算前の位置を返す。
  探索なし、ロックなし、競合なし — わずか数命令です。
* **解放（free）** = *何もしない*。解放されたメモリは個別には回収されません。
* **リセット** = スレッドごとのオフセットをゼロに戻し、そのスレッドのスクラッチ領域全体を
  一挙に回収します。

free が no-op であるため、2 回のリセットの間に確保したすべてのバッファは次のリセットまで生存
し続けます。したがってアリーナは **汎用ヒープではありません** — それは *1 処理単位のための
スクラッチパッド* です。1 つの処理単位の確保をすべて行い、その後 `mpc_cuda_arena_reset()` が
次のためにきれいに消去します。

### 6.2 メモリ配置

アリーナは 1 つの連続した `cudaMalloc` ブロックで、*常駐* スレッドごとに固定サイズの **slab**
に分割されます。

```
mpc_cuda_arena_base ─┐
                     ▼
   ┌──────────┬──────────┬──────────┬─────  ...  ─────┐
   │ thread 0 │ thread 1 │ thread 2 │                 │   各領域 = SLAB バイト
   │  slab    │  slab    │  slab    │                 │
   └──────────┴──────────┴──────────┴─────  ...  ─────┘
        ▲
        └─ スレッドの確保は自分の slab 内で上方向にバンプする。
           mpc_cuda_arena_top[tid] は現在のオフセット（ハイウォーターマーク）
```

各スレッドはグローバルスレッド ID で索引付けされ、ちょうど 1 つの slab を所有するので、
**スレッド間の競合はありません** — スレッドは互いの slab に決して触れません。同じフックが
mini-gmp と（`mpfr-gmp.c` を通じて）MPFR・MPC の *両方* を裏で支えるので、アリーナをインス
トールするとすべての層が一度に高速化されます。

サイズをどう設定しても結果を正しく保つ、2 つの安全フォールバックがあります。

* **アリーナ未インストール**（`mpc_cuda_arena_base == NULL`）→ 確保はデバイスの
  `malloc`/`free` に行きます。正しいが、遅いだけです。これが既存の CPU ホストパスやテストが
  影響を受けない理由です。
* **slab 枯渇**（1 つの処理単位が `SLAB` バイトより多く必要とする）→ *その* 確保だけが
  デバイスの `malloc` にあふれ、通常どおり解放されます。結果は依然として正しく、あふれた部分
  の高速化を失うだけです。したがって超過は静かに起こります — 誤った答えやクラッシュではなく、
  *期待外れの性能* として現れます。

### 6.3 あなたが選ぶ 2 つの数

アリーナの総メモリは、2 つの独立した量の積です。

```
アリーナのバイト数  =  LAUNCH  ×  SLAB
                      （常駐    （スレッド
                       スレッド数） あたりバイト数）
```

* **`LAUNCH` = `LBLOCKS × LTHREADS`** — 起動する *固定* のスレッド数。これは **オキュパンシー
  （占有率）** のために選びます。深い MPFR/MPC の呼び出し連鎖は limb をグローバルメモリの
  slab に保持し、そのメモリレイテンシを隠すには多くのワープが同時に常駐している必要があります。
  数千スレッド（デモは 8K–16K を使用）が問題全体をグリッドストライドする形が正解です — 小さな
  問題に要素ごとに 1 スレッドを割り当てる形ではありません。`LAUNCH` は問題サイズ `N` とは独立
  です。
* **`SLAB`** — スレッドあたりのスクラッチのバイト数。これが正しく設定すべき値で、§6.4 はその
  選び方についてです。

### 6.4 `SLAB` のサイズ設定

`SLAB` は **1 処理単位のピーク同時生存スクラッチ** を保持できなければなりません。free が
no-op であるため、「ピーク生存量」は「最後のリセット以降に確保したすべて」に等しくなります —
つまり `SLAB` は、1 要素分の処理が `mpc_cuda_arena_reset()` までに確保する総バイト数です。

その数を左右するのは 2 つの要素です。

1. **精度。** *p* ビットの `mpfr_t` の仮数部は、おおよそ `ceil(p/64) × 8` バイトに小さなヘッダ
   を加えたものです。スクラッチ使用量は精度にほぼ **線形** にスケールします。*p* を倍にすると、
   必要な slab もおおよそ倍になります。
2. **演算の深さ。** 単純な加算は一時値が数個で済みますが、超越関数（`exp`、`log`、`sin`）や
   正しく丸められる `div` は多数を確保します。**複素数（MPC）は実数（MPFR）よりはるかに大食い**
   です。各複素演算は複数の実数演算に展開され、しばしば拡張された内部精度で行われるためです。

**経験的な目安**（デモが出荷時に採用している値、**1024 ビット** において）：

| ワークロード | `SLAB` | 備考 |
|----------|--------|------|
| 実数 MPFR `axpy` / 内積 | **32 KB** | 数個の `mpfr_t` と算術スクラッチ |
| 複素数 MPC `axpy` / 内積 | **256 KB** | 実数の約 8 倍：より深い連鎖、より多くの一時値 |

これらを出発点とし、**精度に線形でスケール** させてください。例えば 4096 ビットの実数 MPFR
カーネルなら、`32 KB × (4096/1024) = 128 KB` 程度から始めます。多めに切り上げてください —
slab を過剰に確保してもメモリを消費するだけですが、過小だと静かに遅いパスへ落ちてしまいます。

### 6.5 正確なピークを実測する（推奨）

推測する必要はありません。`mpc_cuda_arena_top[tid]` は *まさに* そのスレッドのハイウォーター
マークです。`reset` を呼ばずに **1 つ** の処理単位を実行し、その後 `mpc_cuda_arena_top` をホスト
にコピーバックすれば、そのオフセットがその処理単位の消費バイト数そのものです。

```cuda
// 何もあふれないよう、わざと大きめのアリーナ（例：8 MB/スレッド）をインストールし、
// 1 スレッドで 1 要素だけを実行し、reset は呼ばない：
//     ... 1 要素分の mpfr/mpc の処理 ...
// (mpc_cuda_arena_reset() は呼ばない)

size_t peak;
cudaMemcpy(&peak, top, sizeof(size_t), cudaMemcpyDeviceToHost);
printf("peak live scratch = %zu bytes\n", peak);   // SLAB はこれより少し大きく設定する
```

`SLAB` はそのピークに安全マージン（例えば 25–50%）を加えた値に設定し、入力依存のばらつきを
吸収できるようにしてから、扱いやすいサイズに切り上げてください。これにより、サイズ設定が
当て推量から 1 回の計測へと変わります。

### 6.6 総予算への適合と、健全性チェックリスト

`LAUNCH × SLAB` は、入出力データと併せて GPU メモリに収まらなければなりません。デモは
**512 MB**（実数 axpy：16K スレッド × 32 KB）と **2 GB**（複素数 axpy：8K スレッド × 256 KB）
を確保します。通常の順序は、まずオキュパンシーのために `LAUNCH` を選び、次に `SLAB` が少なく
とも実測ピーク以上であることを確認します。もし `LAUNCH × SLAB` がメモリ予算を超えるなら、
slab をピーク未満に切り詰めるのではなく *`LAUNCH`（オキュパンシー）を下げて* ください。

* **アキュムレータをアリーナの外に置く。** リダクションループ（内積、級数）では、アキュムレータ
  はリセットをまたいで生存する必要があります — `MPFR_DECL_INIT`（実数）や `MPC_DECL_INIT`
  パターン（複素数、`demos/matvec_mpc.cu` 参照）で **スタック** に置きます。そうすればアリーナ
  は常に *1 つの* 積和だけを保持するので、ループ長によらず `SLAB` を小さく保てます（これは
  まさに §5 の累積の例が行っていることです）。
* **処理単位ごとに 1 回リセット** します。各グリッドストライド反復の先頭で行い、スレッドの
  生存期間ごとではありません。1 つのスレッドは多くの要素を処理しますが、一度に必要なのは
  `SLAB` 分だけです。
* **slab が小さすぎる兆候：** 結果はビット単位で正確なのに、性能がデモの高速化を大きく下回る。
  これは静かな `malloc` フォールバックです。ピークを（§6.5 で）再計測し、`SLAB` を上げて
  ください。
* **アリーナが大きすぎる兆候：** 起動時に `cudaMalloc` が失敗する / メモリ不足。`SLAB` を実測
  ピークに向けて下げるか、`LAUNCH` を下げてください。

これらはすべて、デモをビルドする際に `-D` 上書き（`SLAB`、`LBLOCKS`、`LTHREADS`、`N`、`PREC`）
でライブラリを再コンパイルせずに調整できます。

---

## 7. プログラムのリンク

`libmpc_cuda` はデバイス再配置可能コードなので、プログラムは静的アーカイブに対して
**デバイスリンク** します。カーネルを `-dc` でコンパイルし、`-rdc=true` と `libmpc_cuda.a` で
リンクすると、`nvlink` はあなたのカーネルが参照するデバイスオブジェクトだけを正確に取り込み
ます。方法は 3 通りあります。

**(a) インストール済みの補助** — 最も簡単：

```sh
mpc_cuda-link my_kernel.cu my_program
./my_program
```

アーキテクチャは `CUDA_ARCH=sm_90 mpc_cuda-link …` で上書きできます。また、追加のリンク入力
（例えば CPU リファレンスのオブジェクトとシステムライブラリ）は末尾の引数として、あるいは
`EXTRA_LINK` で渡せます。

**(b) ビルドツリーから** — `tools/link_program.sh my_kernel.cu my_program`（`make` の後）、
または一括の `tools/build_cuda_test.sh my_kernel.cu my_program`（再生成 + コンパイル +
リンク）。

**(c) 手動 / pkg-config** — `-dc` でコンパイルし、アーカイブをデバイスリンクします。

```sh
nvcc -dc -rdc=true $(pkg-config --cflags mpc_cuda) my_kernel.cu -o my_kernel.o
nvcc -rdc=true my_kernel.o $(pkg-config --libs mpc_cuda) -o my_program
```

（`pkg-config --libs mpc_cuda` は `-L$(libdir) -lmpc_cuda` に展開されます。`.a` と `.so` の
両方がインストールされているため、デバイスリンクでは静的アーカイブを直接名指しして
— `$(libdir)/libmpc_cuda.a` — 強制するか、常に `.a` をデバイスリンクする `mpc_cuda-link`
補助を使ってください。`.so` は CPU 上で `cu_*` API を呼ぶホスト専用プログラム向けです。）

### システムライブラリとの共存

GPU 側と CPU 側を **別々の** 翻訳単位としてビルドし、一緒にリンクします。GPU の `.cu` は
`cu_mpfr_*` を使い、CPU の `.cpp` はシステムの `<mpfr.h>` をインクルードして `mpfr_*` を使い、
リンクに `-lmpfr -lgmp` を加えます。

```sh
g++ -c cpu_ref.cpp -o cpu_ref.o                       # システムの libmpfr
EXTRA_LINK="cpu_ref.o -lmpfr -lgmp" mpc_cuda-link gpu_side.cu app
```

完全な例は `demos/coexist_demo.cu` + `demos/coexist_cpu.cpp` を参照してください。

---

## 8. API の範囲

API は上流の GMP/MPFR/MPC API に `cu_` プレフィックスを付けたものです。注目点：

* **mini-gmp（整数）：** `cu_mpz_*`（`init`、`set`、`add`、`sub`、`mul`、`tdiv_qr`、
  `pow_ui`、`get_str`、…）と低レベルの `cu_mpn_*`。
* **MPFR（実数）：** 初期化/代入/算術（`cu_mpfr_init2`、`cu_mpfr_set_d`、
  `cu_mpfr_add/sub/mul/div/sqrt`、…）、定数 `cu_mpfr_const_pi` / `cu_mpfr_const_log2`、
  そして **超越関数** `cu_mpfr_exp`、`expm1`、`log`、`log1p`、`sin`、`cos`、`tan`、`atan`、
  `sinh`、`cosh`、`cbrt`、… — すべて正しく丸められ、ホスト MPFR とビット単位で同一です。
* **MPC（複素数）：** `cu_mpc_init2`、`cu_mpc_set_d_d`、`cu_mpc_add/sub/mul/sqr`、および
  複素初等関数 `cu_mpc_sqrt`、`exp`、`log`、`sin`、`cos`、`tan`、`sinh`、`cosh`、`asin`、
  `acos`、`atan`、…

**丸めモード。** 実行時の `cu_mpfr` / `cu_mpc` API は、**上流の MPFR / MPC とまったく同じ
丸めモードの意味論** を持ちます。すなわち各演算は明示的な丸めモード引数を取り、同じ三値
（ternary）を返します。`cu_mpfr_rnd_t` 列挙は `mpfr_rnd_t` と値単位で一致します — `CU_MPFR_RNDN`
（=0、最近接・偶数丸め）、`CU_MPFR_RNDZ`（0 方向）、`CU_MPFR_RNDU`（+∞ 方向）、`CU_MPFR_RNDD`
（−∞ 方向）、`CU_MPFR_RNDA`（0 から離れる方向）、`CU_MPFR_RNDF`（faithful）。`cu_mpc_rnd_t` も
MPC と同様に実部・虚部のモードを 1 つに詰め込み、`CU_MPC_RNDNN … CU_MPC_RNDAA` の全組み合わせと
`CU_MPC_RND(re,im)` / `CU_MPC_RND_RE` / `CU_MPC_RND_IM` の補助マクロを備えます。CPU とまったく
同じように呼び出しごとにモードを選びます。（`cu_compat.h` 経由ではこれらを素の `MPFR_RNDN` /
`MPC_RNDNN` の綴りでも参照できます。）

対照的に、**固定精度** の型（`cu_freal<PB>` / `cu_fcomplex<PB>`、§8.1）は **最近接偶数丸め
（RNDN）専用** で、丸めモード引数を **取りません**。各演算は `PB` ビットで RNDN に丸めます
（複素数型は各成分を RNDN、すなわち `MPC_RNDNN` で丸めます）。方向丸め（directed rounding）が
必要な場合は実行時の `cu_mpfr` / `cu_mpc` API を使ってください。

**デバイスで正しい除算。** MPFR の汎用除算パスは `nvcc` の下で誤コンパイルされます（最適化器が
3 limb 以上の精度で `inf` を生成します）。mpc_cuda は、（正しさを検証済みの）`mpz` プリミティブ
の上に構築した、自己完結的で正しく丸められる `mpfr_div` を代わりに使います。これはホストの
`mpfr_div` とビット単位で一致し、定数と超越関数の実現を可能にしています。これは透過的です —
あなたは単に `cu_mpfr_div` を呼ぶだけです。

本質的にホスト専用の関数はデバイスでは **利用できません**。書式付き/`*_str` の I/O
（`mpfr_printf`、`mpfr_set_str`、`mpfr_get_str` のパース、`mpz_out_str`、…）と、mini-gmp に
ない `mpf_t`/`mpq_t` 変換です（これらのデバイススタブは呼ばれるとトラップします）。

### 8.1 固定精度の高速パス — `cu_fp::cu_freal<PB>`

精度が **コンパイル時に既知** の場合、アンブレラはレジスタ常駐の固定精度実数型も公開します。
これは実行時の `cu_mpfr` パスよりも劇的に高速でありながら、**MPFR の最近接偶数丸めとビット
単位で一致** します。

```cpp
#include "mpc_cuda.cuh"
using cu_fp::cu_freal;

__global__ void k(const double *x, const double *y, double *out, int n) {
  cu_freal<256> a = 1.5;                       // 256-bit mantissa
  for (int i = ...; i < n; ...) {
    cu_freal<256> r = a * cu_freal<256>(x[i]) + cu_freal<256>(y[i]);
    out[i] = (double) r;
  }
}
```

* **`PB` は仮数部の幅（ビット）** で、**32** の任意の倍数（32、64、96、128、160、256、512、
  1024、2048、…）です。仮数は `ceil(PB/64)` 個の limb に **レジスタ上で** 保持されます — アリーナ
  なし、`cudaLimitMallocHeapSize` なし、実行時精度のディスパッチなし。（非常に深い連鎖に対して
  のみ `cudaLimitStackSize` を設定してください。）
* **演算：** `operator+ - *`、または自由関数 `cu_fp::cu_fmul`、`cu_fadd`、`cu_fsub`。変換は
  `cu_freal<PB>::from_double` / `(double)x`（暗黙の `cu_freal<PB>(d)` と `(double)` キャストを
  提供）。各演算は `PB` ビットで `mpfr_mul`/`mpfr_add`/`mpfr_sub` とまったく同様に RNDN で
  丸めます。
* **ヘッダオンリー** で、**GPU でもホストでも** 利用できます（ホストパスは `__uint128` を使用）。
  この型単体であれば `libmpc_cuda` へのリンクは不要です。
* **なぜ速いのか：** コンパイル時の精度により `ptxas` がすべての limb をレジスタに保持できる
  ため、筆算的な乗算と加算/丸めがグローバルメモリへのトラフィックゼロで実行されます。GB10 で
  1024 ビットの場合、AXPY において実行時の `cu_mpfr` GPU パスより ~**140 倍** 高速です（かつ
  ビット単位で同一）。`PB` = 32…2048 にわたり `mul`/`add`/`sub` についてシステム MPFR と
  ビット単位で一致することを検証済みです。
* **トレードオフ：** 精度はコンパイル時定数でなければなりません。実行時に選ぶ精度や、除算 /
  超越関数には、上記の `cu_mpfr` API を使ってください。

**固定精度の複素数 — `cu_fp::cu_fcomplex<PB>`。** 同じコンパイル時精度の複素数型が `cu_freal`
の上に乗ります。

```cpp
using cu_fp::cu_fcomplex;
cu_fcomplex<256> a(1.5,-0.25), x(1.0,0.5), y(2.0,-1.0);
cu_fcomplex<256> z = a*x + y;            // operators + - *
double re = z.real_d(), im = z.imag_d();
```

複素数乗算は **成分ごとに正しく丸められ** ます — `re = ∘(ar·br − ai·bi)`、
`im = ∘(ar·bi + ai·br)` — これは **厳密な** 2·N limb の部分積から計算されます（レジスタ常駐の
`fmms`/`fmma`、二重丸めなし）ので、**MPC の `MPC_RNDNN` とビット単位で一致** します。`PB` =
32…2048 で検証済みです。GB10 で 1024 ビットの場合、複素数 AXPY `z = a·x + y` は実行時の
`cu_mpc` GPU パスより ~**67 倍** 高速です（ビット単位で同一）。

`make sample-fixed` は `demos/sample_fixed.cu`（実数 **と** 複素数、ヘッダオンリー、`-Iinclude`
のみ）をビルドして実行します。`make check-fixed` は `cu_freal` と `cu_fcomplex` の両方を
システム MPFR/MPC に対してビット単位で一致するか検証します。

**固定精度の初等関数**（`mpc_cuda/cu_fmath.cuh`、`cu_fcmath.cuh`。アンブレラからも取り込まれ
ます）。`cu_freal<PB>` / `cu_fcomplex<PB>` 上の高精度（正しく丸められるわけ **ではない** — 下記
参照）初等関数です。

* 実数：`cu_fp::cu_`{`sqrt`,`cbrt`,`exp`,`expm1`,`log`,`log1p`,`sin`,`cos`,
  `tan`,`atan`,`sinh`,`cosh`,`asin`,`acos`,`atanh`,`pow`,`fdiv`}、加えて定数
  `cu_pi<PB>()`、`cu_ln2<PB>()`。
* 複素数：`cu_fp::cu_`{`cexp`,`clog`,`csqrt`,`csin`,`ccos`,`ctan`,`csinh`,
  `ccosh`,`cdiv`}。

これらは作業精度 `PB + CU_FGUARD`（128 ガードビット）でニュートン反復＋引数簡約された級数に
より計算され、その後 `PB` に丸められます。これは **~PB ビットに忠実で、一般的な範囲では
MPFR/MPC と ≤ ~1 ULP で一致** します — 検証スイートにわたり経験的には **0 ULP** です
（`make check-fmath`、`PB` = 64…1024）。これは意図的に **正しく丸められません**：テーブルメーカー
のジレンマのケースには無制限の Ziv ループが必要であり、レジスタ常駐の固定精度とは相容れない
ためです。最後の 1 ビットまで正しい丸めが必要な場合は実行時の `cu_mpfr`/`cu_mpc` API を使って
ください。実数/複素数の **特殊** 関数（ガンマ、ゼータ、erf、ベッセル、…）は今後の課題です。
低～中精度で最速です（値がレジスタにとどまるため）。非常に高い `PB` ではワーキングセットが
あふれます。

**固定精度の高速パスが有利な領域（`make bench3`）。** GB10 上の AXPY `r = a·x + y`、すべて
ビット単位で正確：

| mantissa bits | `cu_freal` GPU | `cu_mpfr` GPU | CPU MPFR | freal vs CPU |
|---|---|---|---|---|
| 128–1024 | ~0.005–0.008 ms (flat) | 0.05–0.39 ms | 0.19–0.68 ms | up to **83×** |
| 2048 | 0.056 ms (7× jump) | 0.51 ms | 0.32 ms | 5.7× |
| 4096 | 0.184 ms | 1.00 ms | 0.47 ms | 2.6× |
| 8192 | 0.823 ms | 1.99 ms | 0.81 ms | **1.0× (parity)** |

`cu_freal` の時間は 1024 ビットまで横ばいで（完全にレジスタ常駐）、その後 **2048 ビットで ~7 倍
跳ね上がります** — ここでスレッドごとの limb 配列（N = 32 limb）がローカルメモリにあふれます —
そして 8192 ビットで CPU 1 コアと同等に達します。したがって **≤ ~1024 ビット** では固定精度の
パスを選び、非常に高い精度では実行時の `cu_mpfr`/`cu_mpc` パスか CPU を使ってください。複素数側
（`cu_fcomplex` 対 `cu_mpc` 対 CPU MPC）も同じ形を辿り、固定精度のリードはさらに大きくなります
（1024 ビットでピークは `cu_mpc` に対して ~48 倍、CPU に対して ~107 倍）。

> **デバイスのスタックサイズに関する落とし穴。** 深い `cu_mpc`/`cu_mpfr` の呼び出し連鎖
> （`mpc_mul → mpfr_fmms → mpfr_sub → …`）には、引き上げた
> `cudaDeviceSetLimit(cudaLimitStackSize, …)` が必要です。GB10 ではこの上限に最大値があります
> （≈ 数百 KB）：大きすぎる要求（例：512 KB）は **`cudaErrorInvalidValue` で拒否** され、戻り値を
> 確認しないとスタックは静かに ~1 KB のデフォルトのままとなり、カーネルが **スタックオーバー
> フロー** します（「illegal memory access」として現れます）。128 KB は受け入れられ、十分です。
> `cudaDeviceSetLimit` の戻り値を必ず確認してください。

---

## 9. デモとベンチマーク

すべてのデモは GPU 対 CPU の **時間と精度** を比較します（CPU の基準はこのライブラリ自身の
ホストパスであり、すべての結果がビット単位で正確です）。NVIDIA GB10、1024 ビットにて：

| ターゲット | 内容 | 結果 |
|--------|--------------|--------|
| `make axpy-mpfr` / `axpy-mpc` | `y = a·x + y`、実数 / 複素数 | ~40 倍高速、ビット単位で正確 |
| `make matvec-mpfr` / `matvec-mpc` | 行列ベクトル積 `y = A·x` | 最大 ~16 倍 |
| `make matmul-mpfr` / `matmul-mpc` | 行列積 `C = A·B` | **~105 倍** |
| `make bench-mpfr` / `bench-mpc` | 関数ごとの初等/超越関数ベンチマーク | 22–77 倍 |
| `make sample-fixed` | 固定精度 `cu_freal<PB>` / `cu_fcomplex<PB>`（実数 + 複素数） | ヘッダオンリー、ビット単位で正確 |
| `make check-fixed` | `cu_freal`/`cu_fcomplex` をシステム MPFR/MPC と照合、PB = 32…2048 | すべてビット単位で一致 |
| `make check-fmath` | 固定精度初等関数の MPFR/MPC に対する ULP 精度 | ≤1 ULP（スイートでは 0） |
| `make bench3` | AXPY 128..8192 ビット：GPU `cu_freal` 対 GPU `cu_mpfr` 対 CPU MPFR | クロスオーバーのスイープ（上記） |
| `make coexist` | 1 バイナリ内の `cu_mpfr`（GPU）+ システム `libmpfr`（CPU） | π、log 2 が一致 |
| `make cputest` | GPU `cu_mpfr`/`cu_mpc` 対 システム `libmpfr`/`libmpc` | ビット単位で正確、すべて一致 |
| `make cpubench` | 同じスイートを計時：関数ごとに GPU 対 システム CPU | 高速化 + ビット単位で正確 |

`make cputest` / `make cpubench` は、このライブラリの GPU 移植と通常の CPU `libmpfr`/`libmpc`
を 1 つのバイナリにリンクし（`cu_` 名前空間により共存できます）、GPU を CPU と照合します。
これらにはシステムライブラリとその `<mpfr.h>`/`<mpc.h>` ヘッダが必要です。CPU 側は
`demos/cpu_ref.cpp` で、両側は型に依存しない `demos/cpu_ref.h` を共有します。

グループターゲット：`make demos`、`make matrix`、`make bench`。ビルドスクリプトを直接呼ぶ
際は、`-D` 上書き（`N`、`PREC`、`LBLOCKS`、`LTHREADS`、`SLAB`）でサイズを調整します。

---

## 10. 動作原理

`tools/cudafy_minigmp.py`、`cudafy_mpfr.py`、`cudafy_mpc.py` は無改変の上流ソースを読み、CUDA
から呼べる版を出力します。デバイスから到達可能なすべての関数に `__host__ __device__` を付け、
ホスト専用関数（stdio/`realloc`/ctype）を検出してホスト専用のままにし、読み取り専用テーブルに
アドレス空間マクロを付け、スレッド状態と定数キャッシュをデバイスセーフにし、`realloc` を
malloc+copy+free でエミュレートします。次に `tools/cu_prefix.py` がエクスポートシンボルを
`cu_` 名前空間にリネームします。パイプライン全体が再実行可能なので、新しい上流リリースは
スクリプトを再実行することで再適応できます。

`tools/build_lib.sh` は生成された各ソースを一度コンパイルします。`tools/link_program.sh`
（およびインストール済みの `mpc_cuda-link`）は、それらのオブジェクトに対してユーザープログラム
のデバイスリンク閉包を解決します。

---

## 11. 制限事項

* **共有される指数/フラグ状態。** MPFR の指数範囲とフラグ（`__gmpfr_emin/emax/flags`）は
  スレッド間で共有されます。テストしたスレッド数までは結果がホストと 1e-13 以内で一致します
  が、完全にレースフリーな *スレッドごと* の状態は今後の課題です。
* **デバイス I/O なし。** 書式付き出力と文字列パースはホスト専用です。
* **`mpf_t`/`mpq_t`** 変換はありません（mini-gmp ビルド）。そのスタブはトラップします。
* **ビルドの並行性。** 同じツリーに対して複数のビルドを同時に実行しないでください
  （共有の `build/` ディレクトリ）。
* これは初期の **0.0.1** リリースです。インターフェースやパッケージングは変わりうります。

---

## 12. ライセンス

mpc_cuda は **GNU 劣等一般公衆利用許諾書（GNU Lesser General Public License）バージョン 3
または（あなたの選択により）それ以降の任意のバージョン**（LGPL-3.0-or-later）の下で配布され
ます。これは、変換・同梱している上流の GNU MPFR および MPC と整合しています。`COPYING.LESSER`
（LGPLv3）および `COPYING`（GPLv3）を参照してください。

多倍長演算は上流の GNU MP、MPFR、MPC プロジェクトの成果です。`AUTHORS` を参照してください。
