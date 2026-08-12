#!/usr/bin/env bash
# Applique le nettoyage du repo DC-Lite.
# Usage : depuis la racine de ton clone local du repo,
#   bash setup_repo.sh
#
# Le script ecrit les fichiers, puis cree six commits separes.
# Il ne pousse rien : relis avec `git log -p` avant `git push`.

set -euo pipefail

if [ ! -d .git ]; then
  echo "Erreur : lance ce script depuis la racine du repo git." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Erreur : working tree non propre. Commite ou stash d'abord." >&2
  exit 1
fi

echo "Branche : $(git rev-parse --abbrev-ref HEAD)"
echo

echo "[1/6] Remove Colab artifacts and hardcoded credentials from the notebook"
cat > pipeline.ipynb <<'__DCLITE_PIPELINE_IPYNB__'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DC-Lite: serialization, training and evaluation pipeline\n",
    "\n",
    "Run on Colab with an A100. `ROOT` points at the working directory holding\n",
    "the architecture files and the serialized LOBSTER streams; set it to a local\n",
    "path to run outside Colab. LOBSTER data is licensed and is not included.\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "ROOT = \"/content/drive/MyDrive/hnet_training\"\n",
    "ARCH_DIR = f\"{ROOT}/architecture\"\n",
    "DATA_DIR = f\"{ROOT}/data_complete/data_serialized\"\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "doIrRT6LWyZS"
   },
   "source": [
    "#Mount Drive\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6TIbZSKbC4vQ"
   },
   "source": [
    "#Requirements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 429
    },
    "executionInfo": {
     "elapsed": 10210,
     "status": "error",
     "timestamp": 1757364212322,
     "user": {
      "displayName": "Felix Ls",
      "userId": "05203630886245940472"
     },
     "user_tz": 240
    },
    "id": "yidxpl9aWGkj",
    "outputId": "428d4028-0add-4988-92cd-464ad4f681c0"
   },
   "outputs": [],
   "source": [
    "!pip install SciencePlots\n",
    "import matplotlib.pyplot as plt\n",
    "import os, json, math, scienceplots, glob, torch, datetime, sys, numpy, subprocess, textwrap, shutil\n",
    "\n",
    "from datetime import datetime\n",
    "plt.style.use(['science', 'grid'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BK5pXbgmUgLU"
   },
   "source": [
    "#Serialization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "069SOi4qTp7j"
   },
   "source": [
    "##Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 37350,
     "status": "ok",
     "timestamp": 1756930327663,
     "user": {
      "displayName": "Felix Ls",
      "userId": "05203630886245940472"
     },
     "user_tz": 240
    },
    "id": "LMDuATmwTnpq",
    "outputId": "ea7e92be-8a3b-4615-8f3e-fd9c0fc1e5a0"
   },
   "outputs": [],
   "source": [
    "!python /content/drive/MyDrive/hnet_training/architecture/serialize_lobster.py \\\n",
    "  --csv /content/drive/MyDrive/hnet_training/data_fast_hyperparameter/training/merged_training_fast.csv \\\n",
    "  --outdir /content/drive/MyDrive/hnet_training/data_fast_hyperparameter/data_serialized/ \\\n",
    "  --schemes all \\\n",
    "  --no_header"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8fSmx2MvTtaT"
   },
   "source": [
    "##Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 6879,
     "status": "ok",
     "timestamp": 1756930334549,
     "user": {
      "displayName": "Felix Ls",
      "userId": "05203630886245940472"
     },
     "user_tz": 240
    },
    "id": "EBr9hPUhDwJQ",
    "outputId": "079383a3-cf09-4255-ef46-3173eaa7a401"
   },
   "outputs": [],
   "source": [
    "!python /content/drive/MyDrive/hnet_training/architecture/serialize_lobster.py \\\n",
    "  --csv /content/drive/MyDrive/hnet_training/data_fast_hyperparameter/validation/merged_validation_fast.csv \\\n",
    "  --outdir /content/drive/MyDrive/hnet_training/data_fast_hyperparameter/data_serialized/ \\\n",
    "  --schemes all \\\n",
    "  --no_header"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "nwzSAHa4LA8l"
   },
   "source": [
    "##Test Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 7040,
     "status": "ok",
     "timestamp": 1756930195545,
     "user": {
      "displayName": "Felix Ls",
      "userId": "05203630886245940472"
     },
     "user_tz": 240
    },
    "id": "QhqU41VCMo7e",
    "outputId": "20a8958e-4ec3-4754-d9f4-fc1c9105c38d"
   },
   "outputs": [],
   "source": [
    "!python /content/drive/MyDrive/hnet_training/architecture/serialize_lobster.py \\\n",
    "  --csv /content/drive/MyDrive/hnet_training/data_complete/test/merged_test.csv \\\n",
    "  --outdir /content/drive/MyDrive/hnet_training/data_complete/data_serialized/ \\\n",
    "  --schemes all \\\n",
    "  --no_header"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7y2dJ_IGnIu_"
   },
   "source": [
    "##How many bytes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "puxhmf54nKlY"
   },
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "streams = {\n",
    "    \"bit_packed\": {\n",
    "        \"train\": [\"/content/drive/MyDrive/hnet_training/data/messages_training.bit_packed.bin\"],\n",
    "        \"val\":   [\"/content/drive/MyDrive/hnet_training/data/messages_validation.bit_packed.bin\"],\n",
    "        \"rec_size\": 21,\n",
    "    },\n",
    "    \"byte_aligned\": {\n",
    "        \"train\": [\"/content/drive/MyDrive/hnet_training/data/messages_training.byte_aligned.bin\"],\n",
    "        \"val\":   [\"/content/drive/MyDrive/hnet_training/data/messages_validation.byte_aligned.bin\"],\n",
    "        \"rec_size\": 22,\n",
    "    },\n",
    "    \"utf8_delim\": {\n",
    "        \"train\": [\"/content/drive/MyDrive/hnet_training/data/messages_training.utf8_delim.bin\"],\n",
    "        \"val\":   [\"/content/drive/MyDrive/hnet_training/data/messages_validation.utf8_delim.bin\"],\n",
    "        \"rec_size\": None,\n",
    "    },\n",
    "}\n",
    "\n",
    "def total_bytes(paths):\n",
    "    return sum(os.path.getsize(p) for p in paths)\n",
    "\n",
    "def count_lines(path):\n",
    "\n",
    "    cnt = 0\n",
    "    with open(path, \"rb\") as f:\n",
    "        for block in iter(lambda: f.read(1024*1024), b\"\"):\n",
    "            cnt += block.count(b\"\\n\")\n",
    "    return cnt\n",
    "\n",
    "for name, spec in streams.items():\n",
    "    tb = total_bytes(spec[\"train\"])\n",
    "    vb = total_bytes(spec[\"val\"])\n",
    "    print(f\"\\n== {name} ==\")\n",
    "    print(f\"train bytes: {tb}  ({tb/1e6:.3f} MB)\")\n",
    "    print(f\"val   bytes: {vb}  ({vb/1e6:.3f} MB)\")\n",
    "    if spec[\"rec_size\"] is not None:\n",
    "        print(f\"train records \u2248 {tb // spec['rec_size']}\")\n",
    "        print(f\"val   records \u2248 {vb // spec['rec_size']}\")\n",
    "    else:\n",
    "        tr = sum(count_lines(p) for p in spec[\"train\"])\n",
    "        vr = sum(count_lines(p) for p in spec[\"val\"])\n",
    "        print(f\"train records (lines): {tr}\")\n",
    "        print(f\"val   records (lines): {vr}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xCPAe0WSTwll"
   },
   "source": [
    "##Sanity Check"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "VhJAJZ8XJWn5"
   },
   "outputs": [],
   "source": [
    "!ls -lh /content/bytes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "tIQINhmlJkS9"
   },
   "outputs": [],
   "source": [
    "import os, math\n",
    "paths = [\n",
    "  (\"/content/bytes/messages.byte_aligned.bin\", 22),\n",
    "  (\"/content/bytes/messages.bit_packed.bin\",   21),\n",
    "]\n",
    "for p,rec in paths:\n",
    "    n = os.path.getsize(p)\n",
    "    print(f\"{os.path.basename(p)}: size={n}  records\u2248{n//rec}  remainder={n%rec}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "5KIti00iJsFC"
   },
   "outputs": [],
   "source": [
    "import struct\n",
    "S = struct.Struct(\"<Q B I I i B\")\n",
    "with open(\"/content/bytes/messages.byte_aligned.bin\",\"rb\") as f:\n",
    "    for i in range(3):\n",
    "        b = f.read(22)\n",
    "        if len(b) < 22: break\n",
    "        t_ns, et, oid, size, price, dir_u8 = S.unpack(b)\n",
    "        direction = 1 if dir_u8==1 else -1\n",
    "        print(i, dict(t_ns=t_ns, EventType=et, OrderID=oid, Size=size, Price=price, Direction=direction))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "2tRcGIGDKL32"
   },
   "outputs": [],
   "source": [
    "import struct\n",
    "S = struct.Struct(\"<Q I I i B\")\n",
    "with open(\"/content/bytes/messages.bit_packed.bin\",\"rb\") as f:\n",
    "    for i in range(3):\n",
    "        b = f.read(21)\n",
    "        if len(b) < 21: break\n",
    "        t_ns, oid, size, price, packed = S.unpack(b)\n",
    "        et = packed & 0b111\n",
    "        direction = 1 if ((packed>>3)&1)==0 else -1\n",
    "        print(i, dict(t_ns=t_ns, EventType=et, OrderID=oid, Size=size, Price=price, Direction=direction))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "2ZxjxxPWKVZO"
   },
   "outputs": [],
   "source": [
    "!head -n 3 /content/bytes/messages.utf8_delim.bin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "DxLRH_WFKZVX"
   },
   "outputs": [],
   "source": [
    "!xxd -l 64 -g 1 /content/bytes/messages.byte_aligned.bin | head -n 3\n",
    "!xxd -l 64 -g 1 /content/bytes/messages.bit_packed.bin   | head -n 3\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "a_pjIHVkp6ua"
   },
   "source": [
    "#Size of the model D-Lite"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 77,
     "status": "ok",
     "timestamp": 1756854199420,
     "user": {
      "displayName": "Felix Ls",
      "userId": "05203630886245940472"
     },
     "user_tz": 240
    },
    "id": "kjpFBRW9p8VD",
    "outputId": "6ebc285d-5a07-44de-b2fd-ceadecda96da"
   },
   "outputs": [],
   "source": [
    "ARCH_DIR = \"/content/drive/MyDrive/hnet_training/architecture\"\n",
    "print(\"ARCH_DIR exists:\", os.path.exists(ARCH_DIR))\n",
    "print(\"ARCH_DIR contents:\", os.listdir(ARCH_DIR))\n",
    "\n",
    "if ARCH_DIR not in sys.path:\n",
    "    sys.path.insert(0, ARCH_DIR)\n",
    "\n",
    "from dc_lite import DCLiteLM\n",
    "m = DCLiteLM(d_model_tok=256, d_model_chunk=384,\n",
    "             n_layers_tok=2, n_heads_tok=4,\n",
    "             n_layers_chunk=4, n_heads_chunk=6)\n",
    "total = sum(p.numel() for p in m.parameters())\n",
    "print(f\"{total:,} parameters  (~{total/1e6:.2f}M)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kkhEw02sUnoq"
   },
   "source": [
    "#Training"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fbc7V4raDptU"
   },
   "source": [
    "##Fast Random Search: Hyperparameters Search"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JdqkiHiwUhWg"
   },
   "source": [
    "##Training of the full model once hyperparameters are found\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "background_save": true,
     "base_uri": "https://localhost:8080/"
    },
    "id": "3BJw9Sj3YWal",
    "outputId": "28ea0e8f-684f-42bc-d51d-fa1a98bf705c"
   },
   "outputs": [],
   "source": [
    "ARCH_DIR = \"/content/drive/MyDrive/hnet_training/architecture\"\n",
    "DATA_DIR = \"/content/drive/MyDrive/hnet_training/data_complete/data_serialized\"\n",
    "\n",
    "CACHE_TRAIN = f\"{DATA_DIR}/merged_training.bit_packed.bin\"\n",
    "CACHE_VAL   = f\"{DATA_DIR}/merged_validation.bit_packed.bin\"\n",
    "CACHE_TEST  = f\"{DATA_DIR}/merged_test.bit_packed.bin\"\n",
    "\n",
    "HP = dict(\n",
    "    seq_len=1024,\n",
    "    batch_size=64,\n",
    "    epochs=24,\n",
    "    lr=0.0024,\n",
    "    wd=0.01,\n",
    "    dropout=0.30,\n",
    "    target_chunk_len=64,\n",
    "    aux_w=0.03,\n",
    "    tau=0.60,\n",
    "    accum=1,\n",
    "    grad_clip=1.0,\n",
    "    early_stop_patience=3,\n",
    "    early_stop_min_delta=0.002,\n",
    "    amp=True,\n",
    "    num_workers=4,\n",
    ")\n",
    "\n",
    "SERIAL = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
    "OUTDIR = f\"{ARCH_DIR}/runs/full_{SERIAL}\"\n",
    "os.makedirs(OUTDIR, exist_ok=True)\n",
    "\n",
    "for p in (CACHE_TRAIN, CACHE_VAL):\n",
    "    if not os.path.exists(p):\n",
    "        raise FileNotFoundError(f\"Missing serialized file: {p}\")\n",
    "if not os.path.exists(f\"{ARCH_DIR}/train_dc_lite.py\"):\n",
    "    raise FileNotFoundError(f\"Missing training script: {ARCH_DIR}/train_dc_lite.py\")\n",
    "\n",
    "print(\"Training will write to:\", OUTDIR)\n",
    "print(\"Train file:\", CACHE_TRAIN)\n",
    "print(\"Val   file:\", CACHE_VAL)\n",
    "\n",
    "amp_flag = \"--amp\" if HP[\"amp\"] else \"\"\n",
    "\n",
    "# --- Launch training ---\n",
    "!python \"{ARCH_DIR}/train_dc_lite.py\" \\\n",
    "  --train_files \"{CACHE_TRAIN}\" \\\n",
    "  --val_files   \"{CACHE_VAL}\" \\\n",
    "  --seq_len {HP['seq_len']} \\\n",
    "  --batch_size {HP['batch_size']} \\\n",
    "  --epochs {HP['epochs']} \\\n",
    "  --lr {HP['lr']} \\\n",
    "  --wd {HP['wd']} \\\n",
    "  --dropout {HP['dropout']} \\\n",
    "  --target_chunk_len {HP['target_chunk_len']} \\\n",
    "  --aux_w {HP['aux_w']} \\\n",
    "  --tau {HP['tau']} \\\n",
    "  --accum {HP['accum']} \\\n",
    "  --grad_clip {HP['grad_clip']} \\\n",
    "  --early_stop_patience {HP['early_stop_patience']} \\\n",
    "  --early_stop_min_delta {HP['early_stop_min_delta']} \\\n",
    "  --num_workers {HP['num_workers']} \\\n",
    "  --save_last --save_every 1 \\\n",
    "  {amp_flag} \\\n",
    "  --resume \"/content/resume.pt\" \\\n",
    "  --outdir \"{OUTDIR}\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "t4PTXSu6EqoR"
   },
   "source": [
    "###Graph from the trained model -use the last checkpoint:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Rb2ydDKfolas"
   },
   "outputs": [],
   "source": [
    "runs = [\n",
    "  (\"/content/drive/MyDrive/hnet_training/architecture/runs/dc_lite_bit/history.json\",   \"bit-packed\"),\n",
    "  (\"/content/drive/MyDrive/hnet_training/architecture/runs/dc_lite_align/history.json\", \"byte-aligned\"),\n",
    "  (\"/content/drive/MyDrive/hnet_training/architecture/runs/dc_lite_utf8/history.json\",  \"utf8+delim\"),\n",
    "]\n",
    "\n",
    "plt.figure(figsize=(7,5))\n",
    "for path,label in runs:\n",
    "    with open(path) as f:\n",
    "        h = json.load(f)\n",
    "    y = h[\"val_ppl\"]\n",
    "    x = list(range(len(y)))\n",
    "    plt.plot(x, y, label=label)\n",
    "plt.title(\"Validation Perplexity vs. Epoch (Three Serialization Schemes)\")\n",
    "plt.xlabel(\"epoch\")\n",
    "plt.ylabel(\"perplexity\")\n",
    "plt.grid(True, which=\"both\", linestyle=\"--\", alpha=0.5)\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"/content/compare_val_ppl.png\", dpi=180)\n",
    "print(\"Saved /content/compare_val_ppl.png\")\n",
    "\n",
    "plt.figure(figsize=(7,5))\n",
    "for path,label in runs:\n",
    "    with open(path) as f:\n",
    "        h = json.load(f)\n",
    "    y = h[\"val_loss\"]\n",
    "    x = list(range(len(y)))\n",
    "    plt.plot(x, y, label=label)\n",
    "plt.title(\"Validation Loss vs. Epoch (Three Serialization Schemes)\")\n",
    "plt.xlabel(\"epoch\")\n",
    "plt.ylabel(\"avg CE loss (nats/byte)\")\n",
    "plt.grid(True, which=\"both\", linestyle=\"--\", alpha=0.5)\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"/content/compare_val_loss.png\", dpi=180)\n",
    "print(\"Saved /content/compare_val_loss.png\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0qipM0avE_xl"
   },
   "source": [
    "###Generic code for each serialization scheme\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "msLU7ZkcY0MM",
    "outputId": "b89a8fe3-cbcb-4bca-8f27-d68065343294"
   },
   "outputs": [],
   "source": [
    "ARCH_DIR = \"/content/drive/MyDrive/hnet_training/architecture\"\n",
    "DATA_DIR = \"/content/drive/MyDrive/hnet_training/data\"\n",
    "OUT_DIR  = f\"{ARCH_DIR}/runs/dc_lite\"\n",
    "\n",
    "src = f\"{ARCH_DIR}/dc_lite_py.py\"\n",
    "dst = f\"{ARCH_DIR}/dc_lite.py\"\n",
    "\n",
    "if os.path.exists(src) and not os.path.exists(dst):\n",
    "    shutil.move(src, dst)\n",
    "print(\"Architecture files:\", os.listdir(ARCH_DIR))\n",
    "\n",
    "!nvidia-smi || echo\n",
    "\n",
    "!python \"/content/drive/MyDrive/hnet_training/architecture/train_dc_lite.py\" \\\n",
    "  --train_files \"/content/drive/MyDrive/hnet_training/data/messages_training.bit_packed.bin\" \\\n",
    "  --val_files   \"/content/drive/MyDrive/hnet_training/data/messages_validation.bit_packed.bin\" \\\n",
    "  --seq_len 2048 --batch_size 32 --epochs 12 --amp \\\n",
    "  --lr 0.0015 --wd 0.05 --dropout 0.30 \\\n",
    "  --target_chunk_len 64 --aux_w 0.05 --tau 0.70 \\\n",
    "  --early_stop_patience 3 --early_stop_min_delta 0.002 \\\n",
    "  --outdir \"/content/drive/MyDrive/hnet_training/architecture/runs/dc_lite_bit\"\n",
    "\n",
    "# # Byte-aligned (22B/rec)\n",
    "# !python \"/content/drive/MyDrive/hnet_training/architecture/train_dc_lite.py\" \\\n",
    "#   --train_files \"/content/drive/MyDrive/hnet_training/data/messages_training.byte_aligned.bin\" \\\n",
    "#   --val_files   \"/content/drive/MyDrive/hnet_training/data/messages_validation.byte_aligned.bin\" \\\n",
    "#   --seq_len 2048 --batch_size 32 --epochs 12 --amp \\\n",
    "#   --lr 0.0015 --wd 0.05 --dropout 0.30 \\\n",
    "#   --target_chunk_len 64 --aux_w 0.05 --tau 0.70 \\\n",
    "#   --early_stop_patience 3 --early_stop_min_delta 0.002 \\\n",
    "#   --outdir \"/content/drive/MyDrive/hnet_training/architecture/runs/dc_lite_align\"\n",
    "\n",
    "# # UTF-8 + delimiter\n",
    "# !python \"/content/drive/MyDrive/hnet_training/architecture/train_dc_lite.py\" \\\n",
    "#   --train_files \"/content/drive/MyDrive/hnet_training/data/messages_training.utf8_delim.bin\" \\\n",
    "#   --val_files   \"/content/drive/MyDrive/hnet_training/data/messages_validation.utf8_delim.bin\" \\\n",
    "#   --seq_len 2048 --batch_size 32 --epochs 12 --amp \\\n",
    "#   --lr 0.0015 --wd 0.05 --dropout 0.30 \\\n",
    "#   --target_chunk_len 64 --aux_w 0.05 --tau 0.70 \\\n",
    "#   --early_stop_patience 3 --early_stop_min_delta 0.002 \\\n",
    "#   --outdir \"/content/drive/MyDrive/hnet_training/architecture/runs/dc_lite_utf8\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JbZVyVhsvzQU"
   },
   "source": [
    "##Resume Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "0s8UhPjEv1Z5",
    "outputId": "95c4bb2e-ab59-4145-c1ed-1c60f9d63c44"
   },
   "outputs": [],
   "source": [
    "!python \"/content/drive/MyDrive/hnet_training/architecture/train_dc_lite.py\" \\\n",
    "  --train_files \"/content/drive/MyDrive/hnet_training/data/messages_training.bit_packed.bin\" \\\n",
    "  --val_files   \"/content/drive/MyDrive/hnet_training/data/messages_validation.bit_packed.bin\" \\\n",
    "  --seq_len 2048 --batch_size 32 --epochs 12 --amp \\\n",
    "  --lr 0.0015 --wd 0.05 --dropout 0.30 \\\n",
    "  --target_chunk_len 64 --aux_w 0.05 --tau 0.70 \\\n",
    "  --early_stop_patience 3 --early_stop_min_delta 0.002 \\\n",
    "  --resume \"/content/drive/MyDrive/hnet_training/architecture/runs/dc_lite_bit/best.pt\" \\\n",
    "  --save_last --save_every 1 \\\n",
    "  --outdir \"/content/drive/MyDrive/hnet_training/architecture/runs/dc_lite_bit\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FqDYkeZUgj5s"
   },
   "source": [
    "##Downstream tasks"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lFL7BtbHgncr"
   },
   "source": [
    "###Distribution comparison and Pnl Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "qynqL3Ig6Vxx"
   },
   "outputs": [],
   "source": [
    "# --- CONFIG ---\n",
    "ARCH_DIR = \"/content/drive/MyDrive/hnet_training/architecture\"\n",
    "CKPT = f\"{ARCH_DIR}/runs/full_*/best.pt\"\n",
    "# Choose one of your serialized files (validation or test)\n",
    "DATA_FILE = \"/content/drive/MyDrive/hnet_training/data_complete/data_serialized/merged_validation.byte_aligned.bin\"\n",
    "# If you only have bit_packed or utf8_delim, set that path instead:\n",
    "# DATA_FILE = \"/content/.../merged_validation.bit_packed.bin\"\n",
    "# DATA_FILE = \"/content/.../merged_validation.utf8_delim.bin\"\n",
    "\n",
    "SEQ_LEN = 1024\n",
    "BATCH_SIZE = 64\n",
    "DEVICE = \"cuda\"\n",
    "\n",
    "# Optional: orderbook-derived midprice CSV (same time base as messages) to compute future returns.\n",
    "# If you don't have it yet, leave as None and PnL will be skipped.\n",
    "MIDPRICE_CSV = None  # e.g. \"/content/drive/.../midprice_validation.csv\"\n",
    "MIDPRICE_TIME_COL = \"time\"       # seconds-after-midnight\n",
    "MIDPRICE_PRICE_COL = \"midprice\"  # midprice\n",
    "\n",
    "# --- IMPORTS ---\n",
    "import os, glob, io, math, json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from collections import Counter\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm import tqdm\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "# --- MODEL (import your DCLite) ---\n",
    "import importlib.util, sys\n",
    "def _import_py(name, path):\n",
    "    spec = importlib.util.spec_from_file_location(name, os.path.join(ARCH_DIR, path))\n",
    "    mod = importlib.util.module_from_spec(spec)\n",
    "    sys.modules[name] = mod\n",
    "    spec.loader.exec_module(mod)\n",
    "    return mod\n",
    "\n",
    "dc_lite = _import_py(\"dc_lite\", \"dc_lite.py\")      # contains DCLiteLM\n",
    "train_mod = _import_py(\"train_dc_lite\", \"train_dc_lite.py\")  # just to reuse helpers if needed\n",
    "\n",
    "# --- DATASET HELPERS ----------------------------------------------------------\n",
    "class ByteDataset(Dataset):\n",
    "    \"\"\"\n",
    "    Minimal byte-level dataset reading a serialized .bin file as raw uint8 stream.\n",
    "    It yields (x, y) where y are next-bytes (next-token) targets.\n",
    "    Works for any of your *.bin variants, but for \"utf8_delim\" it's also easy\n",
    "    to detect field separators (commas/newlines).\n",
    "    \"\"\"\n",
    "    def __init__(self, path, seq_len=1024, stride=None):\n",
    "        self.seq_len = seq_len\n",
    "        self.stride = stride or seq_len\n",
    "        with open(path, \"rb\") as f:\n",
    "            self.buf = np.frombuffer(f.read(), dtype=np.uint8)\n",
    "        # cut last token for x, first token for y\n",
    "        self.N = (len(self.buf) - 1 - seq_len) // self.stride + 1\n",
    "    def __len__(self):\n",
    "        return max(0, self.N)\n",
    "    def __getitem__(self, i):\n",
    "        start = i*self.stride\n",
    "        x = self.buf[start:start+self.seq_len].astype(np.int64)\n",
    "        y = self.buf[start+1:start+self.seq_len+1].astype(np.int64)\n",
    "        return torch.from_numpy(x), torch.from_numpy(y)\n",
    "\n",
    "# --- LOAD DATA ---\n",
    "ds = ByteDataset(DATA_FILE, seq_len=SEQ_LEN, stride=SEQ_LEN)\n",
    "dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)\n",
    "\n",
    "# --- LOAD BEST CHECKPOINT ---\n",
    "def load_best(ckpt_glob):\n",
    "    paths = sorted(glob.glob(ckpt_glob))\n",
    "    if not paths:\n",
    "        raise FileNotFoundError(f\"No checkpoint found for pattern: {ckpt_glob}\")\n",
    "    # choose the latest folder that contains best.pt\n",
    "    bests = []\n",
    "    for root in paths:\n",
    "        if os.path.isdir(root):\n",
    "            bp = os.path.join(root, \"best.pt\")\n",
    "            if os.path.exists(bp):\n",
    "                bests.append(bp)\n",
    "        elif root.endswith(\"best.pt\"):\n",
    "            bests.append(root)\n",
    "    if not bests:\n",
    "        raise FileNotFoundError(f\"No best.pt found under: {ckpt_glob}\")\n",
    "    return sorted(bests)[-1]\n",
    "\n",
    "best_path = load_best(CKPT)\n",
    "print(\"Loading:\", best_path)\n",
    "\n",
    "ckpt = torch.load(best_path, map_location=\"cpu\")\n",
    "model_cfg = ckpt.get(\"model_cfg\", {})  # if your training saved it\n",
    "model = dc_lite.DCLiteLM(**{\n",
    "    # sane defaults; override by saved config if present:\n",
    "    \"vocab_size\": 256,\n",
    "    \"d_model_tok\": model_cfg.get(\"d_model_tok\", 256),\n",
    "    \"d_model_chunk\": model_cfg.get(\"d_model_chunk\", 384),\n",
    "    \"n_layers_tok\": model_cfg.get(\"n_layers_tok\", 2),\n",
    "    \"n_heads_tok\": model_cfg.get(\"n_heads_tok\", 4),\n",
    "    \"n_layers_chunk\": model_cfg.get(\"n_layers_chunk\", 4),\n",
    "    \"n_heads_chunk\": model_cfg.get(\"n_heads_chunk\", 6),\n",
    "    \"mlp_mult\": model_cfg.get(\"mlp_mult\", 2.0),\n",
    "    \"dropout\": model_cfg.get(\"dropout\", 0.3),\n",
    "    \"target_chunk_len\": model_cfg.get(\"target_chunk_len\", 64),\n",
    "    \"boundary_rate_weight\": model_cfg.get(\"boundary_rate_weight\", 0.03),\n",
    "    \"smooth_tau\": model_cfg.get(\"smooth_tau\", 0.6),\n",
    "})\n",
    "model.load_state_dict(ckpt[\"model\"])\n",
    "model.to(DEVICE).eval()\n",
    "\n",
    "# --- EVALUATION: collect true/pred byte histograms ----------------------------\n",
    "true_counts = Counter()\n",
    "pred_counts = Counter()\n",
    "\n",
    "@torch.no_grad()\n",
    "def eval_stream():\n",
    "    for x, y in tqdm(dl, total=len(dl)):\n",
    "        x = x.to(DEVICE, non_blocking=True)\n",
    "        y = y.to(DEVICE, non_blocking=True)\n",
    "        logits, _, _ = model(x, return_aux=True)\n",
    "        pred = torch.argmax(logits, dim=-1)  # [B,T]\n",
    "        # Accumulate histograms over bytes\n",
    "        for t in y.flatten().tolist():\n",
    "            true_counts[int(t)] += 1\n",
    "        for p in pred.flatten().tolist():\n",
    "            pred_counts[int(p)] += 1\n",
    "\n",
    "eval_stream()\n",
    "\n",
    "# --- Utility: discrete divergence (KL and Jensen-Shannon) ---------------------\n",
    "def to_prob(counts):\n",
    "    total = sum(counts.values())\n",
    "    p = np.zeros(256, dtype=np.float64)\n",
    "    if total > 0:\n",
    "        for k, v in counts.items():\n",
    "            p[int(k)] = v / total\n",
    "    return p\n",
    "\n",
    "def kl_div(p, q, eps=1e-12):  # KL(p||q)\n",
    "    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)\n",
    "    return float(np.sum(p * np.log(p / q)))\n",
    "\n",
    "def js_div(p, q, eps=1e-12):\n",
    "    m = 0.5*(p+q)\n",
    "    return 0.5*kl_div(p, m, eps) + 0.5*kl_div(q, m, eps)\n",
    "\n",
    "P_true = to_prob(true_counts)\n",
    "P_pred = to_prob(pred_counts)\n",
    "print(\"Global byte-level KL(true||pred):\", kl_div(P_true, P_pred))\n",
    "print(\"Global byte-level JS:\", js_div(P_true, P_pred))\n",
    "\n",
    "# --- Focused metrics: Event Type (1..7) and Direction (-1/+1) -----------------\n",
    "# We try BOTH encodings:\n",
    "#   (A) numeric bytes 1..7, 255/\u2026 (for bit_packed), and -1/+1 often show up as 255? varies.\n",
    "#   (B) ASCII digits '1'..'7' -> 49..55 and '-' -> 45 (for utf8_delim).\n",
    "# We'll count both and report whichever mass is non-negligible.\n",
    "\n",
    "def extract_event_type_hist(counts):\n",
    "    # numeric 1..7\n",
    "    evt_numeric = np.array([counts.get(i, 0) for i in range(1,8)], dtype=np.float64)\n",
    "    # ASCII '1'..'7'\n",
    "    evt_ascii = np.array([counts.get(i, 0) for i in range(49,56)], dtype=np.float64)\n",
    "    return evt_numeric, evt_ascii\n",
    "\n",
    "def extract_direction_hist(counts):\n",
    "    # numeric: -1/+1 impossible as bytes; sometimes 1 is used for buy and 255 for signed? unknown in bit_packed.\n",
    "    # ASCII: '-' (45) followed by '1' (49) for -1; and '1'(49) alone for +1 (depending on delim format).\n",
    "    # As a simple proxy, we read counts at ASCII '-' and '1'.\n",
    "    dir_ascii = np.array([counts.get(45, 0), counts.get(49, 0)], dtype=np.float64)  # [-, +]\n",
    "    return dir_ascii\n",
    "\n",
    "def norm_hist(h):\n",
    "    s = h.sum()\n",
    "    return h/s if s>0 else h\n",
    "\n",
    "evt_true_num, evt_true_asc = extract_event_type_hist(true_counts)\n",
    "evt_pred_num, evt_pred_asc = extract_event_type_hist(pred_counts)\n",
    "\n",
    "# Choose the encoding with more mass\n",
    "use_ascii_evt = (evt_true_asc.sum() + evt_pred_asc.sum()) > (evt_true_num.sum() + evt_pred_num.sum())\n",
    "if use_ascii_evt:\n",
    "    Ht_evt, Hp_evt = norm_hist(evt_true_asc), norm_hist(evt_pred_asc)\n",
    "    evt_labels = [str(i) for i in range(1,8)]\n",
    "    title_evt = \"Event Type (ASCII digits '1'..'7')\"\n",
    "else:\n",
    "    Ht_evt, Hp_evt = norm_hist(evt_true_num), norm_hist(evt_pred_num)\n",
    "    evt_labels = [str(i) for i in range(1,8)]\n",
    "    title_evt = \"Event Type (numeric bytes 1..7)\"\n",
    "\n",
    "print(f\"[Event Type] KL(true||pred)={kl_div(Ht_evt, Hp_evt):.4f}, JS={js_div(Ht_evt, Hp_evt):.4f}\")\n",
    "\n",
    "dir_true = norm_hist(extract_direction_hist(true_counts))\n",
    "dir_pred = norm_hist(extract_direction_hist(pred_counts))\n",
    "print(f\"[Direction -/+ (ASCII proxy)] KL(true||pred)={kl_div(dir_true, dir_pred):.4f}, JS={js_div(dir_true, dir_pred):.4f}\")\n",
    "\n",
    "# --- Plots: histograms of EventType and Direction -----------------------------\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def plot_bar_comp(labels, p_true, p_pred, title, fname):\n",
    "    x = np.arange(len(labels))\n",
    "    w = 0.38\n",
    "    plt.figure(figsize=(7,4.5))\n",
    "    plt.bar(x-w/2, p_true, width=w, label=\"true\")\n",
    "    plt.bar(x+w/2, p_pred, width=w, label=\"pred\")\n",
    "    plt.xticks(x, labels)\n",
    "    plt.ylabel(\"probability\")\n",
    "    plt.title(title)\n",
    "    plt.legend()\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(fname, dpi=150)\n",
    "    plt.show()\n",
    "\n",
    "plot_bar_comp(evt_labels, Ht_evt, Hp_evt, title_evt, \"event_type_dist.png\")\n",
    "plot_bar_comp([\"-1\",\" +1 (proxy)\"], dir_true, dir_pred, \"Direction (ASCII proxy)\", \"direction_dist.png\")\n",
    "\n",
    "# --- Optional: PnL from forecast sign and future log-returns ------------------\n",
    "def compute_future_log_return(df, horizon):\n",
    "    \"\"\"\n",
    "    df must contain columns: time, midprice (float)\n",
    "    returns aligned vectors: times[:-h], fret (future log return over horizon)\n",
    "    \"\"\"\n",
    "    p = df[MIDPRICE_PRICE_COL].astype(float).values\n",
    "    fret = np.log(p[horizon:] / p[:-horizon])\n",
    "    t = df[MIDPRICE_TIME_COL].values[:-horizon]\n",
    "    return t, fret\n",
    "\n",
    "def forecast_sign_from_bytes(pred_counts_local=None):\n",
    "    \"\"\"\n",
    "    VERY SIMPLE PROXY:\n",
    "      sign(fcst_t) is approximated by the difference between probabilities of direction '+' vs '-'\n",
    "      measured on the predicted stream around direction tokens (ASCII proxy).\n",
    "    For a stronger estimate you would decode rows and pick the Direction field only.\n",
    "    \"\"\"\n",
    "    if pred_counts_local is None:\n",
    "        pred_counts_local = pred_counts\n",
    "    p_minus = pred_counts_local.get(45, 0)  # '-'\n",
    "    p_plus  = pred_counts_local.get(49, 0)  # '1' (used in '+1')\n",
    "    s = p_plus - p_minus\n",
    "    return 1.0 if s >= 0 else -1.0\n",
    "\n",
    "def pnl_from_fcst_and_fret(fcst_sign, fret):\n",
    "    return fcst_sign * fret\n",
    "\n",
    "if MIDPRICE_CSV and os.path.exists(MIDPRICE_CSV):\n",
    "    df_mid = pd.read_csv(MIDPRICE_CSV)\n",
    "    horizons = [1,5,10,20]\n",
    "    fcst_s = forecast_sign_from_bytes()  # scalar sign; if you can align per-timestep, replace by a vector\n",
    "    for h in horizons:\n",
    "        times, fret = compute_future_log_return(df_mid, h)\n",
    "        pnl = pnl_from_fcst_and_fret(fcst_s, fret)\n",
    "        print(f\"h={h:>2d}  meanPnL={pnl.mean(): .5e}  Sharpe={pnl.mean()/pnl.std(): .3f}  n={len(pnl)}\")\n",
    "else:\n",
    "    print(\"PnL skipped (no midprice file provided). Supply MIDPRICE_CSV to compute PnL.\")\n"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "6TIbZSKbC4vQ",
    "BK5pXbgmUgLU",
    "069SOi4qTp7j",
    "8fSmx2MvTtaT",
    "nwzSAHa4LA8l",
    "7y2dJ_IGnIu_",
    "xCPAe0WSTwll",
    "a_pjIHVkp6ua"
   ],
   "machine_shape": "hm",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
__DCLITE_PIPELINE_IPYNB__
git add -A pipeline.ipynb
git commit -q -m "Remove Colab artifacts and hardcoded credentials from the notebook"

echo "[2/6] Vectorise EMA smoothing, chunk pooling and chunk-length extraction"
cat > dc_lite.py <<'__DCLITE_DC_LITE_PY__'
"""DC-Lite: a byte-level autoregressive model with learned chunking.

The model reads raw bytes (vocab = 256) and learns where message boundaries
fall instead of relying on a fixed tokenizer. A router emits a boundary
probability per position; boundaries are binarised in the forward pass and
differentiated through with a straight-through estimator, so the segmentation
is trained jointly with the encoder.

Pipeline: byte embedding -> encoder -> router -> chunk pooling -> chunk
transformer -> fusion with the byte-level states -> next-byte head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 256


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: causal attention then a feed-forward MLP."""

    def __init__(
        self, d_model: int, n_heads: int, mlp_mult: float = 2.0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        hidden = int(mlp_mult * d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)


def straight_through_step(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Hard threshold in the forward pass, sigmoid gradient in the backward pass."""
    p = torch.sigmoid(logits)
    return (p > threshold).to(p.dtype) - p.detach() + p


def causal_ema(x: torch.Tensor, tau: float, tol: float = 1e-7) -> torch.Tensor:
    """Causal exponential moving average along time, computed as a convolution.

    Implements y[0] = x[0] and y[t] = tau * y[t-1] + (1 - tau) * x[t], which
    expands to y[t] = (1 - tau) * sum_k tau^(t-k) x[k] + tau^(t+1) x[0]. The
    kernel is truncated where tau^L falls below ``tol``, so the cost is a single
    parallel conv1d rather than T sequential steps.

    Args:
        x: tensor of shape [B, T, C].
        tau: smoothing coefficient in [0, 1).
        tol: truncation tolerance on the kernel tail.
    """
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must lie in [0, 1), got {tau}")
    B, T, C = x.shape
    if tau == 0.0:
        return x.clone()

    kernel_len = min(T, max(1, int(math.ceil(math.log(tol) / math.log(tau)))))
    decay = tau ** torch.arange(kernel_len, device=x.device, dtype=x.dtype)

    # conv1d correlates, so the kernel is reversed to make the filter causal.
    weight = decay.flip(0).view(1, 1, kernel_len).expand(C, 1, kernel_len)
    padded = F.pad(x.transpose(1, 2), (kernel_len - 1, 0))
    y = F.conv1d(padded, weight, groups=C).transpose(1, 2)

    # Boundary term carrying the y[0] = x[0] initialisation.
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    init = (tau ** (t + 1)).view(1, T, 1) * x[:, :1]
    return (1.0 - tau) * y + init


def chunk_lengths(boundaries: torch.Tensor) -> torch.Tensor:
    """Chunk lengths implied by a boundary mask, flattened across the batch.

    Args:
        boundaries: bool tensor [B, T], True where a chunk ends.

    Returns:
        1-D int64 tensor of chunk lengths (the trailing partial chunk of each
        row is included).
    """
    B, T = boundaries.shape
    closed = boundaries.clone()
    closed[:, -1] = True  # close the trailing chunk of every row
    idx = closed.nonzero(as_tuple=False)
    rows, cols = idx[:, 0], idx[:, 1]
    prev = torch.full_like(cols, -1)
    prev[1:] = cols[:-1]
    row_start = torch.ones_like(rows, dtype=torch.bool)
    row_start[1:] = rows[1:] != rows[:-1]
    prev = torch.where(row_start, torch.full_like(prev, -1), prev)
    return cols - prev


@dataclass
class DCLiteConfig:
    """Hyperparameters of the DC-Lite model.

    Attributes:
        d_model_byte: width of the byte-level encoder and decoder.
        d_model_chunk: width of the chunk-level transformer.
        target_chunk_len: chunk length the boundary-rate penalty pulls towards.
        boundary_rate_weight: weight of that penalty in the total loss.
        smooth_tau: EMA coefficient applied to the router logits.
    """

    d_model_byte: int = 256
    d_model_chunk: int = 384
    n_layers_byte: int = 2
    n_heads_byte: int = 4
    n_layers_chunk: int = 4
    n_heads_chunk: int = 6
    n_layers_decoder: int = 1
    mlp_mult: float = 2.0
    dropout: float = 0.1
    target_chunk_len: int = 64
    boundary_rate_weight: float = 0.05
    smooth_tau: float = 0.6
    boundary_threshold: float = 0.5


class DCLiteLM(nn.Module):
    """Byte-level language model with a learned, differentiable segmentation."""

    def __init__(self, config: DCLiteConfig | None = None, **overrides) -> None:
        super().__init__()
        self.config = config or DCLiteConfig(**overrides)
        cfg = self.config

        self.byte_embed = nn.Embedding(VOCAB_SIZE, cfg.d_model_byte)
        self.pos_byte = PositionalEncoding(cfg.d_model_byte)
        self.encoder = nn.ModuleList(
            TransformerBlock(cfg.d_model_byte, cfg.n_heads_byte, cfg.mlp_mult, cfg.dropout)
            for _ in range(cfg.n_layers_byte)
        )
        self.encoder_ln = nn.LayerNorm(cfg.d_model_byte)

        self.router = nn.Sequential(
            nn.Linear(cfg.d_model_byte, 128), nn.GELU(), nn.Linear(128, 1)
        )

        self.chunk_in = nn.Linear(cfg.d_model_byte, cfg.d_model_chunk)
        self.pos_chunk = PositionalEncoding(cfg.d_model_chunk)
        self.chunk_blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model_chunk, cfg.n_heads_chunk, cfg.mlp_mult, cfg.dropout)
            for _ in range(cfg.n_layers_chunk)
        )
        self.chunk_ln = nn.LayerNorm(cfg.d_model_chunk)

        self.fuse = nn.Linear(cfg.d_model_byte + cfg.d_model_chunk, cfg.d_model_byte)
        self.decoder = nn.ModuleList(
            TransformerBlock(cfg.d_model_byte, cfg.n_heads_byte, cfg.mlp_mult, cfg.dropout)
            for _ in range(cfg.n_layers_decoder)
        )
        self.decoder_ln = nn.LayerNorm(cfg.d_model_byte)
        self.head = nn.Linear(cfg.d_model_byte, VOCAB_SIZE, bias=False)

    def segment(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route byte states to chunk ids.

        Returns:
            boundaries: [B, T] soft-hard boundary indicators in {0, 1}.
            chunk_ids: [B, T] int64 chunk index of each position.
        """
        logits = causal_ema(self.router(h), tau=self.config.smooth_tau)
        boundaries = straight_through_step(logits, self.config.boundary_threshold)
        boundaries = boundaries.squeeze(-1)
        # A chunk always opens at t=0; done out of place to keep autograd happy.
        opening = torch.ones_like(boundaries[:, :1])
        boundaries = torch.cat([opening, boundaries[:, 1:]], dim=1)
        chunk_ids = (boundaries.cumsum(dim=1) - 1.0).long().clamp_min_(0)
        return boundaries, chunk_ids

    @staticmethod
    def pool_chunks(h: torch.Tensor, chunk_ids: torch.Tensor, n_chunks: int) -> torch.Tensor:
        """Mean-pool byte states within each chunk.

        Args:
            h: [B, T, C] byte-level states.
            chunk_ids: [B, T] chunk index per position.
            n_chunks: padded number of chunks.

        Returns:
            [B, n_chunks, C] chunk representations, zero where the row has fewer
            chunks than ``n_chunks``.
        """
        B, T, C = h.shape
        offsets = torch.arange(B, device=h.device).unsqueeze(1) * n_chunks
        flat_ids = (chunk_ids + offsets).reshape(-1)
        sums = h.new_zeros(B * n_chunks, C).index_add_(0, flat_ids, h.reshape(-1, C))
        counts = h.new_zeros(B * n_chunks, 1).index_add_(
            0, flat_ids, h.new_ones(B * T, 1)
        )
        return (sums / counts.clamp_min(1.0)).view(B, n_chunks, C)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_chunk_lengths: bool = False,
    ):
        """Predict the next byte at every position.

        Args:
            x: [B, T] int64 byte ids in [0, 255].
            return_aux: also return the boundary-rate penalty and routing stats.
            return_chunk_lengths: include a tensor of chunk lengths in the stats.

        Returns:
            ``logits`` of shape [B, T, 256], or ``(logits, aux_loss, stats)``
            when ``return_aux`` is set.
        """
        B, T = x.shape
        h = self.pos_byte(self.byte_embed(x))
        for block in self.encoder:
            h = block(h)
        h = self.encoder_ln(h)

        boundaries, chunk_ids = self.segment(h)
        n_chunks = int(chunk_ids.max().item()) + 1  # single host sync per step

        c = self.chunk_in(self.pool_chunks(h, chunk_ids, n_chunks))
        c = self.pos_chunk(c)
        for block in self.chunk_blocks:
            c = block(c)
        c = self.chunk_ln(c)

        # Broadcast each chunk state back to the positions it covers.
        gather_idx = chunk_ids.unsqueeze(-1).expand(-1, -1, c.size(-1))
        chunk_per_byte = c.gather(1, gather_idx)

        y = self.fuse(torch.cat([h, chunk_per_byte], dim=-1))
        for block in self.decoder:
            y = block(y)
        logits = self.head(self.decoder_ln(y))

        if not return_aux:
            return logits

        n_boundaries = boundaries.sum(dim=1)
        rate = n_boundaries / T
        target_rate = 1.0 / self.config.target_chunk_len
        aux_loss = self.config.boundary_rate_weight * ((rate - target_rate) ** 2).mean()

        stats = {
            "avg_chunk_len": T / n_boundaries.mean().clamp_min(1.0).item(),
            "avg_boundaries": n_boundaries.mean().item(),
            "boundary_rate": rate.mean().item(),
        }
        if return_chunk_lengths:
            with torch.no_grad():
                stats["chunk_lengths"] = chunk_lengths(boundaries > 0.5)
        return logits, aux_loss, stats

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
__DCLITE_DC_LITE_PY__
git add -A dc_lite.py
git commit -q -m "Vectorise EMA smoothing, chunk pooling and chunk-length extraction"

echo "[3/6] Track cross-entropy and boundary penalty separately in evaluation"
cat > train_dc_lite.py <<'__DCLITE_TRAIN_DC_LITE_PY__'
"""Pretraining loop for DC-Lite on serialized LOBSTER byte streams.

Example:
    python train_dc_lite.py \
        --train-files data/train.bit_packed.bin \
        --val-files data/val.bit_packed.bin \
        --seq-len 1024 --batch-size 64 --epochs 24 --amp

Cross-entropy is reported in nats per byte; perplexity and bits per byte are
derived from it. The boundary-rate penalty is tracked separately so that the
language-modelling numbers stay comparable across runs with different penalty
weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dc_lite import VOCAB_SIZE, DCLiteConfig, DCLiteLM  # noqa: E402


class ByteSequenceDataset(Dataset):
    """Fixed-length windows over a concatenated stream of raw bytes.

    Args:
        paths: files to concatenate, in order.
        seq_len: window length in bytes.
        stride: step between window starts; defaults to ``seq_len`` (no overlap).
        bytes_cap: keep only the first N bytes of the stream (0 = keep all).
    """

    def __init__(
        self,
        paths: list[str],
        seq_len: int = 2048,
        stride: int | None = None,
        bytes_cap: int = 0,
    ) -> None:
        raw = b"".join(Path(p).read_bytes() for p in paths)
        if bytes_cap > 0:
            raw = raw[:bytes_cap]
        if len(raw) < seq_len + 1:
            raise ValueError(
                f"stream has {len(raw)} bytes, need at least {seq_len + 1} for seq_len={seq_len}"
            )
        self.data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.starts = torch.arange(0, len(self.data) - seq_len - 1, self.stride)

    def __len__(self) -> int:
        return self.starts.numel()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        return (
            self.data[s : s + self.seq_len].clone(),
            self.data[s + 1 : s + self.seq_len + 1].clone(),
        )


def pick_device() -> torch.device:
    """Best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype_for(device: torch.device, enabled: bool):
    """Half-precision dtype to use for autocast, or None when disabled."""
    if not enabled:
        return None
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype=None,
    collect_chunks: bool = False,
    max_samples: int = 50_000,
) -> dict:
    """Validation pass.

    Returns a dict with cross-entropy (nats/byte), perplexity, bits per byte,
    the boundary-rate penalty, the mean chunk length, and optionally a sample of
    chunk lengths for the histogram.
    """
    model.eval()
    ce_sum, aux_sum, n_bytes, n_batches = 0.0, 0.0, 0, 0
    chunk_len_sum, chunk_lengths = 0.0, []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, aux_loss, stats = model(
                x, return_aux=True, return_chunk_lengths=collect_chunks
            )
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction="sum"
            )
        ce_sum += ce.item()
        aux_sum += aux_loss.item()
        chunk_len_sum += stats["avg_chunk_len"]
        n_bytes += y.numel()
        n_batches += 1

        if collect_chunks and len(chunk_lengths) < max_samples:
            chunk_lengths.extend(stats["chunk_lengths"].tolist())

    ce = ce_sum / n_bytes
    return {
        "ce": ce,
        "ppl": math.exp(ce),
        "bpb": ce / math.log(2.0),
        "aux": aux_sum / n_batches,
        "avg_chunk_len": chunk_len_sum / n_batches,
        "chunk_lengths": chunk_lengths[:max_samples],
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype=None,
    accum_steps: int = 1,
    grad_clip: float = 1.0,
) -> dict:
    """One pass over the training set. Returns mean CE, aux loss and chunk length."""
    model.train()
    ce_sum, aux_sum, n_bytes, n_batches = 0.0, 0.0, 0, 0
    chunk_len_sum = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, aux_loss, stats = model(x, return_aux=True)
            ce = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss = ce + aux_loss

        (loss / accum_steps).backward()
        if step % accum_steps == 0:
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        ce_sum += ce.item() * y.numel()
        aux_sum += aux_loss.item()
        chunk_len_sum += stats["avg_chunk_len"]
        n_bytes += y.numel()
        n_batches += 1

    return {
        "ce": ce_sum / n_bytes,
        "aux": aux_sum / n_batches,
        "avg_chunk_len": chunk_len_sum / n_batches,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data = p.add_argument_group("data")
    data.add_argument("--train-files", nargs="+", required=True)
    data.add_argument("--val-files", nargs="+", required=True)
    data.add_argument("--seq-len", type=int, default=2048)
    data.add_argument("--train-bytes-cap", type=int, default=0,
                      help="truncate the training stream to N bytes (0 = all)")
    data.add_argument("--val-bytes-cap", type=int, default=0)
    data.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))

    model = p.add_argument_group("model")
    model.add_argument("--target-chunk-len", type=int, default=64)
    model.add_argument("--aux-weight", type=float, default=0.05,
                       help="weight of the boundary-rate penalty")
    model.add_argument("--tau", type=float, default=0.6,
                       help="EMA coefficient on the router logits")
    model.add_argument("--dropout", type=float, default=0.1)

    optim = p.add_argument_group("optimisation")
    optim.add_argument("--batch-size", type=int, default=24)
    optim.add_argument("--epochs", type=int, default=8)
    optim.add_argument("--lr", type=float, default=2e-3)
    optim.add_argument("--weight-decay", type=float, default=0.01)
    optim.add_argument("--accum-steps", type=int, default=1)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--amp", action="store_true", help="mixed-precision training")
    optim.add_argument("--patience", type=int, default=0,
                       help="stop after N epochs without validation improvement (0 = off)")
    optim.add_argument("--min-delta", type=float, default=0.0)
    optim.add_argument("--seed", type=int, default=0)

    io = p.add_argument_group("io")
    io.add_argument("--outdir", type=Path, default=Path("runs/dc_lite"))
    io.add_argument("--resume", type=Path, default=None)
    io.add_argument("--save-last", action="store_true")
    io.add_argument("--save-every", type=int, default=0)
    io.add_argument("--collect-chunk-hist", action="store_true")
    io.add_argument("--max-hist-samples", type=int, default=50_000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device()
    amp_dtype = autocast_dtype_for(device, args.amp)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"device={device} amp={amp_dtype}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    loaders = {}
    for split, files, cap in (
        ("train", args.train_files, args.train_bytes_cap),
        ("val", args.val_files, args.val_bytes_cap),
    ):
        ds = ByteSequenceDataset(files, seq_len=args.seq_len, bytes_cap=cap)
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"{split}: {len(ds.data):,} bytes, {len(ds):,} windows")

    config = DCLiteConfig(
        dropout=args.dropout,
        target_chunk_len=args.target_chunk_len,
        boundary_rate_weight=args.aux_weight,
        smooth_tau=args.tau,
    )
    model = DCLiteLM(config).to(device)
    print(f"model: {model.num_parameters():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: dict[str, list] = {
        k: [] for k in
        ("train_ce", "train_aux", "val_ce", "val_ppl", "val_bpb", "val_aux", "avg_chunk_len", "lr")
    }
    best_val = float("inf")
    start_epoch, stale_epochs = 1, 0

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history.update(ckpt.get("history", {}))
        best_val = ckpt.get("best_val", best_val)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch} (best val CE {best_val:.4f})")

    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train = train_one_epoch(
                model, loaders["train"], optimizer, device, amp_dtype,
                args.accum_steps, args.grad_clip,
            )
            val = evaluate(
                model, loaders["val"], device, amp_dtype,
                collect_chunks=args.collect_chunk_hist,
                max_samples=args.max_hist_samples,
            )
            lr = optimizer.param_groups[0]["lr"]

            for key, value in (
                ("train_ce", train["ce"]), ("train_aux", train["aux"]),
                ("val_ce", val["ce"]), ("val_ppl", val["ppl"]),
                ("val_bpb", val["bpb"]), ("val_aux", val["aux"]),
                ("avg_chunk_len", val["avg_chunk_len"]), ("lr", lr),
            ):
                history[key].append(value)
            if args.collect_chunk_hist:
                history["chunk_lengths"] = val["chunk_lengths"]

            print(
                f"epoch {epoch:02d} | lr {lr:.3g} | train CE {train['ce']:.4f} | "
                f"val CE {val['ce']:.4f} | ppl {val['ppl']:.2f} | bpb {val['bpb']:.3f} | "
                f"chunk len {val['avg_chunk_len']:.1f}"
            )

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "history": history,
                "args": {k: str(v) for k, v in vars(args).items()},
                "config": asdict(config),
            }
            if best_val - val["ce"] > args.min_delta:
                best_val = val["ce"]
                checkpoint["best_val"] = best_val
                torch.save(checkpoint, args.outdir / "best.pt")
                stale_epochs = 0
            else:
                stale_epochs += 1
            if args.save_last:
                torch.save(checkpoint, args.outdir / "last.pt")
            if args.save_every and epoch % args.save_every == 0:
                torch.save(checkpoint, args.outdir / f"epoch_{epoch}.pt")

            if args.patience and stale_epochs >= args.patience:
                print(f"early stop at epoch {epoch}: no improvement for {stale_epochs} epochs")
                break
            scheduler.step()
    except KeyboardInterrupt:
        print(f"interrupted at epoch {epoch}")

    with open(args.outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"artifacts written to {args.outdir}")


if __name__ == "__main__":
    main()
__DCLITE_TRAIN_DC_LITE_PY__
git add -A train_dc_lite.py
git commit -q -m "Track cross-entropy and boundary penalty separately in evaluation"

echo "[4/6] Move plotting out of the training script"
cat > plots.py <<'__DCLITE_PLOTS_PY__'
"""Figures from a training run's ``history.json``.

Example:
    python plots.py runs/dc_lite --title "DC-Lite (bit-packed)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=160)
    plt.close(fig)


def plot_history(history: dict, outdir: Path, title: str = "") -> None:
    """Write loss, perplexity, chunk-length and learning-rate figures."""
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["val_ce"]) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, history["train_ce"], label="train")
    ax.plot(epochs, history["val_ce"], label="validation")
    ax.set(xlabel="epoch", ylabel="cross-entropy (nats/byte)", title=f"{title} loss".strip())
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, outdir, "loss.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, history["val_ppl"])
    ax.set(xlabel="epoch", ylabel="perplexity", title=f"{title} validation perplexity".strip())
    ax.grid(alpha=0.3)
    _save(fig, outdir, "perplexity.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, history["avg_chunk_len"])
    ax.set(xlabel="epoch", ylabel="mean chunk length (bytes)",
           title=f"{title} learned chunk length".strip())
    ax.grid(alpha=0.3)
    _save(fig, outdir, "chunk_length.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, history["lr"])
    ax.set(xlabel="epoch", ylabel="learning rate", title=f"{title} schedule".strip())
    ax.grid(alpha=0.3)
    _save(fig, outdir, "learning_rate.png")

    lengths = np.asarray(history.get("chunk_lengths", []), dtype=np.int64)
    if lengths.size:
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
        left.hist(lengths, bins=min(100, max(32, int(np.median(lengths)) * 2)))
        left.set(xlabel="chunk length (bytes)", ylabel="count", title="histogram")
        left.grid(alpha=0.3)
        ordered = np.sort(lengths)
        right.plot(ordered, np.arange(1, ordered.size + 1) / ordered.size)
        right.set(xlabel="chunk length (bytes)", ylabel="ECDF", title="ECDF")
        right.grid(alpha=0.3)
        fig.suptitle(f"{title} chunk-length distribution".strip())
        _save(fig, outdir, "chunk_length_distribution.png")


def compare_runs(runs: dict[str, Path], outdir: Path) -> None:
    """Overlay validation perplexity across serialization schemes."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for label, path in runs.items():
        history = json.loads(Path(path).read_text())
        ax.plot(range(1, len(history["val_ppl"]) + 1), history["val_ppl"], label=label)
    ax.set(xlabel="epoch", ylabel="validation perplexity",
           title="Validation perplexity by serialization scheme")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, outdir, "scheme_comparison.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rundir", type=Path, help="directory containing history.json")
    p.add_argument("--title", default="")
    args = p.parse_args()
    history = json.loads((args.rundir / "history.json").read_text())
    plot_history(history, args.rundir / "figures", args.title)
    print(f"figures written to {args.rundir / 'figures'}")


if __name__ == "__main__":
    main()
__DCLITE_PLOTS_PY__
git add -A plots.py
git commit -q -m "Move plotting out of the training script"

echo "[5/6] Clean up the LOBSTER serializer"
cat > serialize_lobster.py <<'__DCLITE_SERIALIZE_LOBSTER_PY__'
"""Serialize LOBSTER message files into raw byte streams.

Three encodings are provided, so that the effect of the serialization scheme on
the learned segmentation can be measured:

  utf8_delim    UTF-8 text, pipe-separated fields, newline end-of-message.
                Variable length; field widths depend on the values.
  byte_aligned  fixed 22-byte records, little-endian.
  bit_packed    fixed 21-byte records; event type and direction share one byte.

A LOBSTER message row is:
    Time (seconds after midnight), EventType (1-7), OrderID, Size,
    Price (dollars x 10000), Direction (-1 sell, +1 buy)

Example:
    python serialize_lobster.py --csv messages.csv --outdir bytes/ --schemes all
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path
from typing import Callable, Iterator, Sequence

UINT32_MAX = 2**32 - 1
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
DAY_NS = 86_400 * 1_000_000_000

BYTE_ALIGNED = struct.Struct("<QBIIiB")  # t_ns, event, order_id, size, price, direction
BIT_PACKED = struct.Struct("<QIIiB")  # t_ns, order_id, size, price, event|direction


def clamp(x: int, lo: int, hi: int) -> int:
    """Clamp x into [lo, hi]."""
    return max(lo, min(x, hi))


def read_messages(csv_path: Path, has_header: bool) -> Iterator[tuple]:
    """Yield parsed LOBSTER rows, skipping malformed ones.

    Yields:
        (time_sec, event_type, order_id, size, price, direction)
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                yield (
                    float(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                )
            except ValueError:
                continue


def to_nanoseconds(time_sec: float) -> int:
    """Seconds after midnight to integer nanoseconds, clamped to one day."""
    return clamp(round(time_sec * 1_000_000_000), 0, DAY_NS)


def encode_direction(direction: int) -> int:
    """LOBSTER direction (-1 sell, +1 buy) to an unsigned byte (2 sell, 1 buy)."""
    return 1 if direction >= 0 else 2


def pack_event_and_direction(event_type: int, direction: int) -> int:
    """Pack event type into bits 0-2 and direction into bit 3 of a single byte."""
    return (clamp(event_type, 1, 7) & 0b111) | ((encode_direction(direction) - 1) << 3)


def write_utf8_delim(
    csv_path: Path, out_path: Path, has_header: bool, delimiter: str = "|"
) -> int:
    """Write messages as delimited UTF-8 text, one message per line."""
    count = 0
    with open(out_path, "wb") as w:
        for t, event, order_id, size, price, direction in read_messages(csv_path, has_header):
            record = delimiter.join(str(v) for v in (t, event, order_id, size, price, direction))
            w.write(record.encode("utf-8") + b"\n")
            count += 1
    return count


def write_byte_aligned(csv_path: Path, out_path: Path, has_header: bool) -> int:
    """Write fixed-width 22-byte records."""
    count = 0
    with open(out_path, "wb") as w:
        for t, event, order_id, size, price, direction in read_messages(csv_path, has_header):
            w.write(
                BYTE_ALIGNED.pack(
                    to_nanoseconds(t),
                    clamp(event, 1, 7),
                    clamp(order_id, 0, UINT32_MAX),
                    clamp(size, 0, UINT32_MAX),
                    clamp(price, INT32_MIN, INT32_MAX),
                    encode_direction(direction),
                )
            )
            count += 1
    return count


def write_bit_packed(csv_path: Path, out_path: Path, has_header: bool) -> int:
    """Write fixed-width 21-byte records with event type and direction packed."""
    count = 0
    with open(out_path, "wb") as w:
        for t, event, order_id, size, price, direction in read_messages(csv_path, has_header):
            w.write(
                BIT_PACKED.pack(
                    to_nanoseconds(t),
                    clamp(order_id, 0, UINT32_MAX),
                    clamp(size, 0, UINT32_MAX),
                    clamp(price, INT32_MIN, INT32_MAX),
                    pack_event_and_direction(event, direction),
                )
            )
            count += 1
    return count


SCHEMES: dict[str, tuple[str, Callable[[Path, Path, bool], int]]] = {
    "utf8": ("utf8_delim", write_utf8_delim),
    "aligned": ("byte_aligned", write_byte_aligned),
    "packed": ("bit_packed", write_bit_packed),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, required=True, help="LOBSTER message CSV")
    p.add_argument("--outdir", type=Path, required=True, help="directory for the .bin files")
    p.add_argument("--no-header", action="store_true", help="the CSV has no header row")
    p.add_argument("--schemes", nargs="+", default=["all"], choices=["all", *SCHEMES])
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = list(SCHEMES) if "all" in args.schemes else args.schemes
    stem = args.csv.stem

    for key in selected:
        suffix, writer = SCHEMES[key]
        out_path = args.outdir / f"{stem}.{suffix}.bin"
        count = writer(args.csv, out_path, not args.no_header)
        print(f"{suffix}: {count:,} messages -> {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
__DCLITE_SERIALIZE_LOBSTER_PY__
git add -A serialize_lobster.py
git commit -q -m "Clean up the LOBSTER serializer"

echo "[6/6] Add README, license, pinned requirements and a stricter gitignore"
cat > README.md <<'__DCLITE_README_MD__'
# DC-Lite: Tokenizer-Free Byte-Level Modelling of Limit Order Books

A byte-level autoregressive model for limit order book message streams that
learns its own segmentation instead of using a fixed tokenizer or hand-designed
snapshot features.

MSc Statistical Science dissertation, University of Oxford (Distinction, 2025).
Supervisors: Mihai Cucuringu, Stefan Zohren — Oxford-Man Institute of Quantitative Finance.

**[Read the thesis](Delissen_2025_DC_Lite_LOB.pdf)**

---

## Motivation

BPE and its variants are a poor fit for order book data. They are fitted to
character frequency, not to record structure, so numeric fields split
inconsistently depending on their digits, and every token receives the same
compute regardless of how predictable it is. DC-Lite drops the vocabulary
entirely: LOBSTER messages are serialized to raw bytes, and a router learns
where chunk boundaries fall so that predictable prefixes (timestamps, event
codes) are compressed into long chunks while volatile fields (price, size)
receive more capacity.

## Method

A simplified H-Net adapted to order book messages:

1. **Serialization** — LOBSTER rows to raw bytes under three schemes
   (`serialize_lobster.py`), so the effect of field layout on the learned
   segmentation can be measured.
2. **Byte encoder** — embedding over a 256-symbol alphabet plus a shallow causal
   transformer, producing one contextual vector per byte.
3. **Router and dynamic chunking** — an MLP emits a boundary logit per position,
   smoothed by a causal EMA. The logit is thresholded in the forward pass and
   differentiated with a straight-through estimator, so the segmentation trains
   jointly with the rest of the network. A boundary-rate penalty pulls the mean
   chunk length towards a target.
4. **Chunk transformer** — operates on mean-pooled chunk representatives, so the
   main backbone runs on a sequence shorter by roughly the mean chunk length.
5. **Decoder** — chunk states are broadcast back to byte positions, fused with
   the byte-level states, and passed to a next-byte head.

## Serialization schemes

| Scheme | Record size | Layout |
|---|---|---|
| `utf8_delim` | variable | UTF-8 text, pipe-separated, newline end-of-message |
| `byte_aligned` | 22 bytes | little-endian fixed fields |
| `bit_packed` | 21 bytes | as above, event type and direction share one byte |

## Data

LOBSTER Level-3 message files. **No market data is included in this repository**
— LOBSTER data is licensed and must be obtained separately. The serialization
script expects the standard LOBSTER message CSV layout: time, event type, order
ID, size, price (dollars x 10000), direction.

## Usage

```bash
pip install -r requirements.txt

# 1. Serialize messages to byte streams
python serialize_lobster.py --csv data/messages.csv --outdir data/bytes --no-header

# 2. Pretrain
python train_dc_lite.py \
    --train-files data/bytes/train.bit_packed.bin \
    --val-files   data/bytes/val.bit_packed.bin \
    --seq-len 1024 --batch-size 64 --epochs 24 --amp \
    --target-chunk-len 64 --patience 3 --outdir runs/bit_packed

# 3. Figures
python plots.py runs/bit_packed --title "DC-Lite (bit-packed)"
```

Cross-entropy is reported in nats per byte; perplexity and bits per byte follow
from it. The boundary-rate penalty is logged separately so that language-model
numbers stay comparable across runs with different penalty weights.

## Results

| Serialization | Val. bits/byte | Val. perplexity | Mean chunk length |
|---|---|---|---|
| `utf8_delim` | [X] | [X] | [X] |
| `byte_aligned` | [X] | [X] | [X] |
| `bit_packed` | [X] | [X] | [X] |

Main finding: [one sentence].

## Files

```
dc_lite.py             model: byte encoder, router, chunking, chunk transformer, head
serialize_lobster.py   LOBSTER CSV to byte streams (three schemes)
train_dc_lite.py       pretraining loop with early stopping and checkpointing
plots.py               figures from a run's history.json
pipeline.ipynb         end-to-end Colab walkthrough
```

## Limitations and next steps

DC-Lite is pretrained on [scope] and evaluated on next-byte prediction. It is a
pretraining backbone, not a foundation model: establishing that would require
multi-instrument, multi-venue pretraining and evaluation on several unseen
downstream tasks with frozen representations. That is the natural extension of
this work.

## References

- Hwang et al. (2025), *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*
- Pagnoni et al. (2024), *Byte Latent Transformer: Patches Scale Better Than Tokens*
- Zhang, Zohren & Roberts (2019), *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*
- Sirignano & Cont (2019), *Universal features of price formation in financial markets*

## License

MIT — see [LICENSE](LICENSE).
__DCLITE_README_MD__
cat > LICENSE <<'__DCLITE_LICENSE__'
MIT License

Copyright (c) 2025 Felix Delissen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
__DCLITE_LICENSE__
cat > requirements.txt <<'__DCLITE_REQUIREMENTS_TXT__'
numpy>=1.24
torch>=2.0
matplotlib>=3.7
__DCLITE_REQUIREMENTS_TXT__
cat > .gitignore <<'__DCLITE__GITIGNORE__'
# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/

# Training artifacts
runs/
*.pt
*.ckpt

# Data (LOBSTER files are licensed and must not be redistributed)
data/
*.bin
*.csv

# OS
.DS_Store
__DCLITE__GITIGNORE__
git add -A README.md LICENSE requirements.txt .gitignore
git commit -q -m "Add README, license, pinned requirements and a stricter gitignore"

echo
echo "Termine. Six commits crees :"
git log --oneline -6
echo
echo "A faire ensuite, a la main :"
echo "  1. git log -p          # relis les diffs"
echo "  2. git push"
echo "  3. Renommer le repo en dc-lite-lob (Settings > General)"
echo "  4. git remote set-url origin https://github.com/felixdelissen/dc-lite-lob.git"
echo "  5. Ajouter description + topics dans le panneau About"
echo "  6. Ajouter Delissen_2025_DC_Lite_LOB.pdf a la racine"
echo "  7. Remplir les [X] du README"
