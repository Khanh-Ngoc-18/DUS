# Nhận xét kết quả chọn ngưỡng DUS

**Về:** `find_threshold.py` — kết quả T ≈ 0.209, stop_rate 13.2%, miss_rate 2.9%
**Dữ liệu:** `results/dus_per_round.csv` (1135 round / 300 sample / 3 benchmark)
**Script tái lập:** `verify_threshold.py`

---

## 1. Kết quả có tái lập được

Con số khớp chính xác đến từng chữ số: T = 0.2087496, TP/FN/FP/TN = 493/33/492/117,
stop_rate = 0.132159, miss_rate = 0.029075. Code khớp với mô tả trong docstring,
không có lỗi cài đặt. Phần dưới là vấn đề về **cách đặt bài toán**, không phải bug.

---

## 2. Vấn đề nghiêm trọng: quy tắc bị vòng lặp logic

**Quy tắc "dừng khi DUS < T" trùng khít với "dừng khi consensus" — mà vòng debate đã
làm sẵn việc đó.**

Chuỗi bằng chứng:

1. Cả 150 dòng có `DUS < T` đều có `answer_entropy == 0`.
2. `answer_entropy == 0` ⟺ `consensus == True`, khớp 168/168 dòng.
3. Cả 168 dòng consensus đều là **round cuối** của sample đó — vì
   `orchestrator.py:215` có `if consensus and self.early_stop_on_consensus: break`.

Hệ quả: mô phỏng dừng-tuần-tự thật (dừng ở round đầu tiên có DUS < T) cho
**compute tiết kiệm = 0/1135 round = 0%**. Trên 835 round *không phải* round cuối —
nơi duy nhất dừng sớm có thể tiết kiệm gì — số round sẽ dừng là **0**
(min DUS ở nhóm này = 0.208750, đúng bằng T, mà điều kiện là `<` chặt).

Nói cách khác: 13.2% "dừng sớm" không phải compute tiết kiệm được, mà là đếm lại
các round mà debate vốn đã kết thúc. Đây sẽ là câu hỏi đầu tiên của reviewer và
hiện chưa có cách trả lời.

**Cần sửa:** loại toàn bộ round consensus / round cuối khỏi tập quét ngưỡng.
Chỉ 835 round không-cuối mới là ứng viên hợp lệ.

---

## 3. Ngưỡng không robust

`T = 0.2087496` nằm đúng trên một mass point có **149 dòng trùng giá trị**.

| điều kiện | stop_rate | miss_rate |
|---|---|---|
| `dus < T` | 13.2% | 2.9% |
| `dus <= T` | 26.3% | **8.4%** |
| `dus < T + 1e-9` | 26.3% | **8.4%** |

Đổi dấu bất đẳng thức là vỡ ràng buộc 5%. Một điểm tối ưu chỉ tồn tại nhờ dấu `<`
thay vì `<=` thì không nên báo cáo như một ngưỡng vận hành.

Nguyên nhân gốc: DUS chỉ có 43 giá trị rời rạc trên 1135 dòng, vì nó bị chi phối bởi
`answer_entropy` — biến chỉ nhận 3 giá trị khi có 3 agent (0, 0.918, 1.585).

---

## 4. Cách tính tỉ lệ làm phóng đại độ tin cậy

**`miss_rate = FN / n` với n = 1135 round.** Nhưng `final_correct` là nhãn **cấp
sample**, được nhân bản ra mọi round của sample đó. 1135 dòng chỉ đến từ 300 sample,
và riêng 141 sample có 6 round đã đóng góp 846 dòng. Các dòng không độc lập, nên
n = 1135 phóng đại cỡ mẫu khoảng 4 lần.

Tính lại theo sample: **miss_rate = 11.0%**, không phải 2.9%.

**Cần sửa:** báo miss_rate theo sample; bootstrap CI resample theo `sample_id`, không
resample theo dòng.

---

## 5. Ngưỡng chung không đạt ràng buộc trên cả 3 benchmark

| | gsm8k | mmlu | strategyqa |
|---|---|---|---|
| accuracy | 0.76 | 0.56 | 0.56 |
| stop_rate @ T=0.209 | 15.2% | 7.5% | 19.3% |
| miss_rate @ T | 0.86% | 1.72% | **6.85%** |
| AUC (toàn bộ round) | 0.683 | 0.618 | 0.573 |
| AUC (round không-cuối) | 0.595 | 0.576 | **0.532** |

Con số 2.9% tổng thể đạt được là nhờ gsm8k kéo trung bình xuống. strategyqa vượt
ngưỡng 5%. Nếu giữ ràng buộc miss_rate < 5% thì phải chọn ngưỡng riêng cho từng
benchmark, và khi đó strategyqa gần như không dừng được câu nào.

Đáng lo hơn: DUS yếu dần theo độ khó, và trên strategyqa (round không-cuối) AUC = 0.532
≈ đoán ngẫu nhiên — trong khi đây lại chính là benchmark có stop_rate cao nhất.
Đúng chiều ngược với mong muốn: dừng nhiều nhất ở nơi tín hiệu tệ nhất.

---

## 6. DUS chưa thực sự là score đa chiều

Trọng số học được trong `results/dus_weights.json`:

| feature | trọng số | AUC riêng (round không-cuối) |
|---|---|---|
| answer_entropy | 0.708 | 0.542 |
| disagreement_persistence | 0.180 | 0.535 |
| answer_flip_rate | 0.112 | 0.544 |
| confidence_variance | **0.000** | 0.497 |

`confidence_variance` có hệ số logistic gốc là **−0.273** (âm) và đã bị clip về 0.
Việc clip này cần được nêu rõ và biện minh — hệ số âm có thể là tín hiệu thật
(agent tự tin đồng đều nhưng cùng sai), không nhất thiết là nhiễu cần loại.

Với 0.708/1.0 trọng số dồn vào entropy và AUC gần như trùng nhau
(DUS 0.616 vs entropy đơn lẻ 0.604 trên toàn bộ round), hiện chưa có bằng chứng
DUS mang thêm thông tin so với chỉ dùng entropy. Cần một ablation rõ ràng để
biện minh cho việc tổ hợp 4 feature.

---

## 7. Nhãn đang dùng không khớp với quyết định cần đưa ra

Nếu dừng ở round *r*, cái được chốt là **đáp án tại round r**, không phải
`final_answer`. Nhưng nhãn hiện tại là `final_correct` — tức là đang đánh giá một
quyết định bằng kết quả của một quyết định khác.

Dữ liệu để sửa đã có sẵn: `debate_agents_*.csv` có cột `normalized_answer` theo
từng agent từng round. Thử thay nhãn bằng "majority vote tại round r có đúng không",
trên 835 round không-cuối:

|  | final ĐÚNG | final SAI |
|---|---|---|
| **round r ĐÚNG** | 194 (23%) | 99 (12%) |
| **round r SAI** | 227 (27%) | 315 (38%) |

AUC với nhãn đúng = 0.593, so với 0.560 khi dùng `final_correct`.

*(Lưu ý: parser `normalized_answer` cho gsm8k đang lỗi — AUC ra `nan`. Cần sửa
trước khi tin con số này.)*

---

## 8. Thiết kế thí nghiệm

- **Mỗi benchmark chỉ có 1 run duy nhất, không lặp seed.** Chưa tách được biến thiên
  ngẫu nhiên của LLM khỏi hiệu ứng thật. Đây là lỗ hổng lớn nhất nếu định công bố.
  Tối thiểu 3–5 seed.
- **Test split quá nhỏ:** 117 round / ~30 sample. miss_rate 1.71% trên test tương ứng
  đúng 2 dòng — không đủ để kết luận.
- **Không có dữ liệu phản thực.** Vì `early_stop_on_consensus=True`, không biết được
  "nếu debate tiếp thì sao" ở các sample bị cắt sớm. Nên chạy một lượt với
  `early_stop_on_consensus=False`, ép đủ 6 round mọi sample.

---

## 9. Điểm làm tốt

- `split.py` chia theo `sample_id` là đúng — tránh được leakage giữa các round của
  cùng một câu hỏi. Đây là chỗ rất dễ làm sai và đã làm đúng.
- Weights fit trên `--split train` (770 dòng), có ghi lại `split` trong file weights.
- Log đủ chi tiết để làm lại toàn bộ phân tích mà **không cần chạy lại LLM**:
  answer/confidence theo từng agent từng round, `total_tokens`, `ground_truth`.
- Ngưỡng ổn định qua các split (train 14.3%/3.4%, val 10.5%/2.0%, test 12.0%/1.7%).

Hạ tầng thu thập dữ liệu là tốt. Vấn đề nằm ở tầng đánh giá.

---

## 10. Việc cần làm, theo thứ tự ưu tiên

**A. Sửa được ngay, không cần chạy lại LLM**

1. Loại round consensus/round cuối khỏi tập quét ngưỡng — chỉ scan trên 835 round không-cuối.
2. Đổi nhãn sang `correct_at_round_r` (sửa parser gsm8k trước).
3. Đánh giá tuần tự theo sample; báo compute tiết kiệm bằng **token** (`total_tokens`
   có sẵn trong jsonl), không phải bằng số round.
4. Bootstrap CI cho miss_rate, resample theo `sample_id`.
5. Chọn ngưỡng riêng cho từng benchmark; báo cáo cả 3, không gộp.
6. Ablation: DUS 4-feature vs chỉ `answer_entropy`. Nêu rõ việc clip trọng số âm.

**B. Cần chạy lại data**

7. 3–5 seed mỗi benchmark.
8. Một lượt chạy `early_stop_on_consensus=False` để có dữ liệu phản thực đủ 6 round.
9. Tăng số câu — 100/benchmark là quá ít cho split 70/20/10.

**C. Baseline bắt buộc phải so**

Bất kỳ phương pháp dừng sớm nào cũng phải đánh bại được:
(i) dừng khi consensus (baseline hiện tại của repo),
(ii) chạy cứng k = 2 round,
(iii) self-consistency không debate, cùng ngân sách token.

Nếu DUS không thắng nổi ba cái này thì bản thân điều đó cũng là kết quả đáng báo cáo.

---

## 11. Gợi ý định khung lại đóng góp

Khung "early stopping bằng DUS" hiện không đứng vững vì vòng lặp logic ở mục 2, và
sửa vặt không cứu được.

Nhưng bảng ở mục 7 chứa một kết quả thật và chưa cần chạy thêm gì:
**debate cứu được 227 ca nhưng phá hỏng 99 ca đang đúng, và 38% số round tiếp tục là
hoàn toàn vô ích.** Net gain chỉ +128/835 — mong manh hơn nhiều so với những gì
literature về multi-agent debate thường tuyên bố.

Câu hỏi đáng theo đuổi không phải "khi nào dừng debate" mà **"debate có đáng không,
và nó hỏng ở đâu"**. Dự đoán được nhóm 99 ca bị debate làm hỏng có giá trị thực tiễn
cao hơn early stopping, và là một finding tiêu cực nhưng mới.
