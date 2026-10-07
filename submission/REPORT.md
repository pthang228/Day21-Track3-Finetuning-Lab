# Báo cáo Lab 21 — Fine-tuning LLM bằng LoRA

**Họ tên:** Nguyễn Lê Phúc Thắng  
**MSSV:** 2A202602638  
**Ngày chạy:** 07/10/2026  
**Thiết bị:** NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM  
**Cấu hình tier:** `CPU` (dùng profile model 0.8B và context 512; phép tính thực tế chạy CUDA)  
**Base model:** `Qwen/Qwen3.5-0.8B`

## Tóm tắt quyết định

LoRA cải thiện rất mạnh tác vụ triage: target tăng từ 0.495 của base với prompt tối ưu
lên 0.985, format giữ ở 1.000. Tuy nhiên, regression giảm từ 0.5778 xuống 0.0667. Vì
vậy cổng hồi quy kết luận **FAILED** và tôi **không đề xuất deploy** adapter hiện tại.
Kết quả cho thấy fine-tuning đã nội hoá schema triage nhưng đồng thời gây catastrophic
forgetting nghiêm trọng. Bước tiếp theo hợp lý là thêm 1–5% replay data phổ thông và
chạy lại cùng gate, không phải nới ngưỡng hay làm yếu baseline.

## 1. Thiết kế thí nghiệm

Tôi chọn Qwen3.5-0.8B vì GPU chỉ có 4 GB VRAM; lựa chọn này cho phép chạy đủ bốn run,
full eval và cả merge check tại chỗ. Dataset mặc định gồm 250 ticket CSKH tiếng Việt
với đầu ra JSON bốn trường. Split cố định seed 42 tạo 225 mẫu train và 25 mẫu validation;
eval dùng đủ 50 target và 15 regression, không đặt `EVAL_LIMIT`.

| Thuộc tính | Giá trị |
|---|---:|
| Mask | `assistant-only` |
| Epoch / optimizer step | 2 / 58 |
| Batch hiệu dụng | 8 |
| p95 / max token quan sát | 98 / 101 |
| `suggested_max_length` | 256 |
| `max_length` thực chạy | 512 |

Tôi giữ `max_length=512` theo profile tier để có biên an toàn cho thay đổi template và
không cắt response; toàn corpus hiện tại vẫn ngắn hơn nhiều nên quyết định này không làm
mất nhãn. Đây là lựa chọn bảo thủ so với p95 làm tròn lên 256 và được khai báo thay vì
coi 512 là con số đoán.

## 2. Bằng chứng mask và chat template

Template **giữ nguyên** khối `<think>` (`reasoning preserved — safe to train on traces`).
Mask proof có 37/94 token được giám sát, `supervised_fraction=0.3936`; hai assert đều
đúng: `answer_is_supervised=true` và `question_is_masked=true`.

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop",
 "sentiment": "trung_tinh"}<|im_end|>
```

Phần system/user prompt nằm trong `masked_preview`, không nằm trong loss. EOS
`<|im_end|>` nằm trong span được giám sát để model học điểm dừng.

## 3. Mốc được đóng băng trước khi train

| Run | Target | Regression | Format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.5778 | 0.000 | 5397.5 |
| (b) base + optimized prompt | 0.495 | 0.5778 | 1.000 | 1296.5 |
| (c) LoRA fine-tune | **0.985** | **0.0667** | 1.000 | 1791.6 |

Baseline (b) được đo và ghi `baselines_frozen.json` trước khi có adapter. Nó tốt hơn
(a) 0.495 target point, đưa format từ 0 lên 1 và nhanh hơn khoảng 4.2 lần; vì vậy đây là
một mốc thật sự mạnh, không phải prompt yếu để làm đẹp kết quả fine-tune. Tôi không sửa
`OPTIMIZED_PROMPT`; SHA được đóng băng là `719e74d3b6232053`.

## 4. Giải phẫu cấu hình

| Run | Placement | r | Trainable | LR | Loss | Target | Thời gian s | Peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 10,822,656 | 1e-4 | 0.3960 | **0.985** | 557.7 | 1.97 |
| `attn_only` | q,v | 271 | 10,822,656 | 1e-4 | 0.4345 | 0.935 | 362.7 | 1.98 |
| `wrong_lr` | text-linear | 16 | 10,822,656 | 1e-5 | 1.5288 | 0.330 | 522.9 | 1.97 |
| `qlora` | text-linear | 16 | 10,822,656 | 1e-4 | 0.4227 | 0.945 | 583.3 | **1.19** |

### 4.1 Vị trí so với rank

`attn_only` dùng r=271 và có đúng cùng 10,822,656 tham số với `correct`, nên đây là đối
chứng cùng ngân sách. Nó đạt target 0.935, thấp hơn `correct` 0.985; loss 0.4345 cũng cao
hơn 0.3960. Trong lần đo này, mở rộng vị trí sang toàn bộ linear của text decoder có ích
hơn việc dồn rank rất lớn vào q/v. Kết quả không chứng minh rank luôn vô dụng, nhưng nó
chứng minh tăng rank không bù hoàn toàn cho placement hẹp ở tác vụ này.

### 4.2 Learning rate

`wrong_lr` chỉ đổi LR từ 1e-4 xuống 1e-5 và giữ nguyên mọi biến còn lại. Loss cuối tăng
từ 0.3960 lên 1.5288, còn target giảm từ 0.985 xuống 0.330, thậm chí thấp hơn baseline
prompt 0.495. Nếu chỉ nhìn việc loss vẫn giảm theo thời gian, tôi có thể kết luận nhầm
rằng run đang học bình thường; độ lớn cuối và target task cho thấy nó underfit rõ rệt.
Đây là bằng chứng LR theo thang LoRA là đòn bẩy thực, không chỉ là chi tiết tuning nhỏ.

### 4.3 QLoRA

QLoRA giảm peak VRAM từ 1.97 xuống 1.19 GB, tiết kiệm 0.78 GB hay khoảng 39.6%. Đổi lại,
target giảm 0.040, latency tăng từ 1791.6 lên 2121.2 ms/mẫu và training lâu hơn 25.6
giây. Trên card 4 GB, mức tiết kiệm là có giá trị, nhưng model 16-bit vốn đã vừa bộ nhớ
và cho chất lượng tốt hơn. Số đo vì vậy ủng hộ khuyến nghị không dùng QLoRA cho run chính
trên Qwen3.5 khi bf16 LoRA vẫn chạy được.

## 5. Phán quyết bốn nhóm

**Kết quả: FAILED.** Target delta là **+0.490**, regression delta là **-0.5111**, format
là 1.000 và latency là 1791.6 ms/mẫu. Fine-tune thắng baseline mạnh ở đúng tác vụ hẹp,
nhưng đánh đổi gần như toàn bộ khả năng trả lời câu hỏi phổ thông: regression chỉ còn
0.0667 so với 0.5778. Mức giảm này vượt rất xa tolerance 0.020, nên target 0.985 không
đủ để biện minh cho deploy. `valid_trace_rate=0.0` không được dùng để kết luận reasoning
collapse vì corpus train là JSON trần, không chứa reasoning trace, và generation tắt
thinking. Nguyên nhân có khả năng nhất là catastrophic forgetting do toàn bộ 225 mẫu đều
cùng một hành vi triage. Tôi sẽ trộn 1–5% replay data phổ thông, giữ nguyên eval và gate,
rồi chạy lại; nếu regression chưa phục hồi thì nên giữ giải pháp prompt (b) hoặc route
riêng adapter chỉ cho endpoint triage.

## 6. Phân tích định tính, gồm cả ca fine-tune thua

| # | Nhóm | Input rút gọn | Base (b) | Fine-tune | Kết luận |
|---:|---|---|---|---|---|
| 1 | target | Áo khoác gió “Bị lỗi” | 0.50; sai intent và urgency | 0.75; chỉ sai intent | FT cải thiện nhưng vẫn thua nhãn |
| 2 | target | Máy xay “Khi nào có tiền về” | 0.50; sai urgency và sentiment | 0.75; chỉ sai urgency | FT cải thiện nhưng còn nhầm mức khẩn |
| 3 | target | Ốp lưng “Giá bao nhiêu” | 0.50; đoán `van_chuyen/cao` | 1.00; đủ bốn trường đúng | FT thắng rõ |
| 4 | regression | Chúc mừng sinh nhật | 1.00; viết lời chúc phù hợp | 0.00; ép thành JSON triage | **FT thua** |
| 5 | regression | 2 mũ 10 bằng bao nhiêu? | 1.00; trả lời 1024 | 0.00; ép thành JSON triage | **FT thua** |
| 6 | regression | Tác giả Truyện Kiều | 1.00; nêu Nguyễn Du | 0.00; ép thành JSON triage | **FT thua** |

Ba lỗi target đều tập trung ở intent/urgency mơ hồ, trong khi product và sentiment ổn
định hơn. Mẫu chung nghiêm trọng hơn nằm ở regression: adapter áp schema triage lên cả
input ngoài miền. Đây là bằng chứng trực tiếp cho catastrophic forgetting và giải thích
vì sao accuracy target cao không đồng nghĩa model đã an toàn để phục vụ chung.

## 7. Merge và hot-swap

NB6 đạt điểm target 0.985 trước merge và 0.985 sau merge, delta 0.000 trong tolerance
0.01. Một base đã nạp và hot-swap thành công ba adapter: `correct`, `attn_only`, `qlora`.
Điều này xác nhận merge không làm tụt chất lượng đo được và kiến trúc multi-adapter hoạt
động; tuy nhiên nó không thay đổi verdict về regression.

## 8. Kết luận và điều học được

Tôi không nên deploy adapter này như một model đa dụng. Về mặt tác vụ hẹp, fine-tuning
thành công rất rõ: target tăng 49 điểm phần trăm so với một prompt đã tối ưu, format đạt
100%, và merge bảo toàn nguyên điểm số. Tuy nhiên, chính phép đo bốn nhóm cho thấy cái
giá bị che khuất nếu chỉ nhìn train loss hoặc target accuracy: khả năng phổ thông giảm
hơn 51 điểm phần trăm. Nguyên nhân hợp lý là dữ liệu train quá đồng nhất, khiến model
học rằng mọi input đều phải được ép về JSON triage. Mask đúng là điều kiện nền tảng,
learning rate 1e-4 là cần thiết để học đủ nhanh, và placement text-linear thắng
attention-only ở cùng ngân sách; nhưng không nút nào trong ba nút đó tự giải quyết quên
thảm hoạ. QLoRA tiết kiệm gần 40% VRAM nhưng mất thêm 4 điểm target trong khi bf16 đã
vừa GPU, nên cũng không phải lựa chọn chính. Quyết định kỹ thuật phù hợp là thêm replay
data ngoài miền, huấn luyện lại rồi yêu cầu regression gate pass trước khi deploy. Nếu
không thể làm vậy, tôi sẽ giữ base + optimized prompt hoặc giới hạn adapter trong một
endpoint triage được route chặt, tuyệt đối không thay thế model chung.

Ba điều tôi học được:

1. Baseline prompt mạnh có thể biến cả format lẫn latency, nên phải đóng băng trước train.
2. Cùng số tham số không có nghĩa cùng năng lực; vị trí adapter quan trọng hơn rank lớn.
3. Target accuracy cao có thể cùng tồn tại với catastrophic forgetting, nên regression
   gate là điều kiện deploy chứ không phải số phụ.

Nếu có thêm hai giờ, tôi sẽ thêm 1%, 3% và 5% replay data phổ thông, giữ nguyên 58 step,
chọn tỉ lệ nhỏ nhất phục hồi regression trong tolerance mà không làm target giảm đáng kể.

## Artefact sử dụng

Số liệu được đọc từ `results/mask_proof.json`, `template_check.json`, `token_stats.json`,
`baselines_frozen.json`, `runs.csv`, `verdict.json`, `autopsy.json`,
`qualitative_comparison.json`, `qualitative_regression_comparison.json` và
`merge_check.json`. Không có số đánh giá nào được ước lượng hoặc điền tay thay cho run.
