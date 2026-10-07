# Reflection — Lab 21

## 1. Điều làm tôi ngạc nhiên nhất

Fine-tune đạt target 0.985 và format 1.000 nhưng regression chỉ còn 0.0667. Ba ví dụ
regression cho thấy model không chỉ trả lời sai mà còn cố ép câu hỏi phổ thông thành JSON
triage. Khoảng cách giữa “rất tốt trên benchmark hẹp” và “không an toàn để deploy chung”
rõ hơn tôi dự đoán.

## 2. Phần tốn thời gian nhất

Phần tốn thời gian nhất là chạy bốn training run và đánh giá sinh văn bản, không phải
viết LoRA config. Tôi đã dự đoán training sẽ lâu, nhưng baseline ngây thơ cũng chậm vì
model sinh dài và không tuân thủ schema. Prompt tối ưu vừa tăng target vừa giảm latency
từ 5397.5 xuống 1296.5 ms/mẫu là một kết quả đáng chú ý.

## 3. Niềm tin đã thay đổi

Trước lab tôi dễ coi loss giảm và target accuracy tăng là đủ để kết luận fine-tune thành
công. Sau phép đo này, tôi không còn tin một chỉ số tác vụ đơn lẻ có thể quyết định
deploy. Regression gate và ví dụ ngoài miền phải được xem là tiêu chí bắt buộc.

## 4. Tôi dùng AI assistant vào việc gì, và nó có thể sai ở đâu

Tôi dùng AI assistant để rà repo, dựng môi trường CUDA tách biệt, chạy pipeline, theo dõi
artefact và tổng hợp report từ các JSON/CSV thật. Assistant ban đầu ước lượng thời gian
theo tốc độ trung bình, nhưng GPU laptop/WDDM làm tốc độ từng step dao động đáng kể; vì
vậy ETA chỉ là khoảng. Tôi kiểm soát rủi ro bằng cách không điền số liệu trước khi run và
chạy `verify.py` ở cuối.

## 5. Nếu fine-tune cho khách hàng thật

Bước đầu tiên của tôi là chốt một eval set đóng băng gồm target, regression, format và
latency trước khi train. Sau đó tôi xây baseline prompt mạnh và proof mask. Với kết quả
lab này, tôi cũng sẽ chuẩn bị replay data ngoài miền ngay từ đầu thay vì chờ catastrophic
forgetting xuất hiện sau khi huấn luyện.
