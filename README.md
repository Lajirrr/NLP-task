# 基于 Transformer 的英译中机器翻译系统

这个项目使用 PyTorch `nn.Transformer` 实现一个简单的英文到中文机器翻译系统。数据集已经放在 `data/` 下，包含训练集、验证集、测试集和中英文词表。

## 目录结构

```text
NLP/
  code/
    config.py       # 默认路径、特殊 token、模型超参数
    dataset.py      # 数据读取、词表加载、batch padding
    model.py        # PositionalEncoding 和 TransformerTranslator
    train.py        # 训练并保存 best checkpoint
    evaluate.py     # 测试集 loss/perplexity/BLEU 和翻译样例
    translate.py    # 单句英文翻译
  data/
    training.txt
    validation.txt
    testing.txt
    word2int_en.json
    int2word_en.json
    word2int_cn.json
    int2word_cn.json
  tests/
```

## 环境安装

当前 Windows 环境中 PyTorch 已安装在 conda `base` 环境。先在 PowerShell 中激活 `base`：

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
```

如果需要重新安装依赖，在激活 `base` 后运行：

```powershell
python -m pip install -r requirements.txt
```

预训练模型路径会使用 Hugging Face `transformers`、`sentencepiece`、`sacremoses` 和 `hf_xet`。如果只运行从零训练的 Transformer，核心依赖仍然是 PyTorch 和 tqdm；如果要运行 `pretrained_*.py`，请先确保 `requirements.txt` 中的依赖已经安装。

默认训练目标是 NVIDIA GPU/CUDA。如果安装的 PyTorch 不能检测到 CUDA，训练脚本会给出明确提示。只想先检查流程时，可以加 `--device cpu`。

## 数据格式

`training.txt`、`validation.txt`、`testing.txt` 每行是一对句子：

```text
英文 token 序列<TAB>中文 token 序列
```

示例：

```text
he is a teacher .	他 是 老师 。
```

模型会在源端英文后追加 `<EOS>`；目标端中文会构造：

- decoder 输入：`<BOS> + 中文 tokens`
- loss 标签：`中文 tokens + <EOS>`

默认目标端仍使用现有中文词级词表。若训练时加 `--target-level char`，中文目标端会先把空格分隔的中文 token 拼回句子，再按字符切分，例如 `他 是 老师 。` 会变成 `他 是 老 师 。`，可以明显减少中文目标端 `<UNK>`。

## 训练

```powershell
python code/train.py --epochs 20 --batch-size 64
```

常用参数：

```powershell
python code/train.py --epochs 30 --batch-size 128 --lr 0.0001 --device cuda
python code/train.py --epochs 1 --batch-size 16 --device cpu
```

训练中文字符级目标端模型：

```powershell
python code/train.py --target-level char --epochs 50 --batch-size 64 --device cuda --checkpoint-dir checkpoints/char
```

轻量增强版字符级训练会启用 label smoothing、Noam warmup 调度，并保留若干个验证集最优 checkpoint 供后续平均：

```powershell
python code/train.py --target-level char --epochs 80 --batch-size 64 --device cuda --checkpoint-dir checkpoints/char-enhanced --label-smoothing 0.1 --scheduler noam --warmup-steps 4000 --keep-best-checkpoints 5
```

训练过程中会输出每个 epoch 的 train loss、valid loss 和 valid perplexity。验证集 loss 最低的模型会保存到：

```text
checkpoints/best.pt
```

## 测试评估

```powershell
python code/evaluate.py --checkpoint checkpoints/best.pt --num-examples 10
```

默认解码会禁止输出目标端 `<UNK>`，这样样例更可读。如果想复现实验中最原始的 greedy 解码，可以加 `--allow-unk`。

```powershell
python code/evaluate.py --checkpoint checkpoints/best.pt --allow-unk
```

如果更重视翻译质量而不是速度，可以使用 beam search。`--beam-size 4` 在验证集上通常比 greedy 稍好，但完整测试集会明显更慢：

```powershell
python code/evaluate.py --checkpoint checkpoints/best.pt --beam-size 4 --length-penalty 0.6
```

评估字符级目标端 checkpoint 时不需要额外参数，脚本会从 checkpoint 自动恢复目标端粒度：

```powershell
python code/evaluate.py --checkpoint checkpoints/char/best.pt --device cuda --beam-size 4
```

如果训练时保留了多个 `best_epoch_*.pt`，可以先做 checkpoint averaging，再评估平均后的模型：

```powershell
python code/average_checkpoints.py --inputs checkpoints/char-enhanced/best_epoch_*.pt --output checkpoints/char-enhanced/averaged.pt
python code/evaluate.py --checkpoint checkpoints/char-enhanced/averaged.pt --device cuda --beam-size 4 --no-repeat-ngram-size 3
```

`evaluate.py` also supports a lightweight checkpoint ensemble by passing more than one
checkpoint after `--checkpoint`. The checkpoints must use the same source/target
vocabularies:

```powershell
python code/evaluate.py --checkpoint checkpoints/char-enhanced/averaged.pt checkpoints/char-adam98-e80/best.pt --device cuda --beam-size 8 --length-penalty 1.5 --no-repeat-ngram-size 2
```

如果只想快速看 test loss/perplexity，不想每次等待完整测试集 decode：

```powershell
python code/evaluate.py --checkpoint checkpoints/char-enhanced/best.pt --device cuda --loss-only
```

如果只想快速估计 BLEU 和查看少量翻译样例，可以限制 decode 数量：

```powershell
python code/evaluate.py --checkpoint checkpoints/char-enhanced/best.pt --device cuda --beam-size 4 --no-repeat-ngram-size 3 --decode-limit 200
```

评估脚本会输出：

- test loss
- test perplexity
- corpus BLEU
- 若干条英文、参考中文、模型翻译样例

## 单句翻译

输入已经分好词的英文：

```powershell
python code/translate.py --checkpoint checkpoints/best.pt --text "tom is a student ."
```

也可以输入普通英文句子，脚本会做简单小写和标点切分：

```powershell
python code/translate.py --checkpoint checkpoints/best.pt --text "Tom is a student."
```

查看中间 token：

```powershell
python code/translate.py --checkpoint checkpoints/best.pt --text "I'll go." --show-tokens
```

单句翻译时也可以使用 beam search：

```powershell
python code/translate.py --checkpoint checkpoints/best.pt --text "tom is a student ." --beam-size 4
```

翻译字符级目标端 checkpoint：

```powershell
python code/translate.py --checkpoint checkpoints/char/best.pt --text "tom is a student ." --device cuda --beam-size 4
```

若翻译出现明显重复，可以在 beam search 时加 ngram 重复抑制：

```powershell
python code/translate.py --checkpoint checkpoints/char-enhanced/averaged.pt --text "tom is a student ." --device cuda --beam-size 4 --no-repeat-ngram-size 3
```

## 预训练模型路径

当前项目也提供一套独立的 Hugging Face OPUS-MT 路径，默认模型是 `Helsinki-NLP/opus-mt-en-zh`。这些脚本不会覆盖从零训练 Transformer 的 checkpoint。

如果本机存在 `checkpoints/pretrained-opus-base/`，`pretrained_translate.py`、`pretrained_evaluate.py` 和 `pretrained_finetune.py` 会默认优先使用这个本地 base 模型；否则才会回退到 Hugging Face 模型名 `Helsinki-NLP/opus-mt-en-zh` 并尝试联网下载。直接用本地预训练模型翻译：

```powershell
python code/pretrained_translate.py --text "tom is a student ." --device auto --num-beams 4
```

在测试集上快速评估前 20 条：

```powershell
python code/pretrained_evaluate.py --limit 20 --device auto --num-beams 4
```

在完整测试集上评估：

```powershell
python code/pretrained_evaluate.py --device cuda --batch-size 16 --num-beams 4
```

在本项目 18000 条训练集上微调 OPUS-MT，输出目录和原来的 `checkpoints/best.pt`、`checkpoints/char/best.pt` 分开：

```powershell
python code/pretrained_finetune.py --epochs 3 --batch-size 8 --device cuda --output-dir checkpoints/pretrained-opus-en-zh
```

微调后翻译或评估时，把 `--model-name-or-path` 指向保存目录：

```powershell
python code/pretrained_translate.py --model-name-or-path checkpoints/pretrained-opus-en-zh --text "tom is a student ." --device cuda
python code/pretrained_evaluate.py --model-name-or-path checkpoints/pretrained-opus-en-zh --device cuda --num-beams 4
```

## 说明

- 默认使用 greedy decoding，并禁止输出 `<UNK>`；需要允许 `<UNK>` 时可加 `--allow-unk`。
- `translate.py` 和 `evaluate.py` 支持 `--beam-size`，beam search 更慢，但通常能略微提升 BLEU 和减少局部错误。
- `translate.py` 和 `evaluate.py` 支持 `--no-repeat-ngram-size`，可减少“美丽的美丽”这类重复，但过大时可能压制合理重复。
- `evaluate.py` 支持 `--loss-only` 和 `--decode-limit`，适合训练期间快速检查，最终报告 BLEU 时再跑完整测试集。
- BLEU 是项目内轻量实现，不依赖 NLTK。
- 中文输出会去掉 `<PAD>`、`<BOS>`、`<EOS>`，然后直接拼接中文 token。
- 新版 checkpoint 会保存 `target_level` 和目标端词表；旧的词级 checkpoint 仍按原方式加载。
- `pretrained_evaluate.py` 输出的是字符级 BLEU，便于和当前字符级目标端结果做大致对照；预训练模型的 loss/perplexity 来自 Hugging Face tokenizer 的 teacher forcing，不和从零训练模型的词表 loss 完全等价。
- 如果输入英文词不在词表中，会映射到 `<UNK>`。
