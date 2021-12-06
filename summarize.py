from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
if torch.cuda.is_available():       
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-small-vi-summarization")
tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-small-vi-summarization")
model.to(device)

# src = "Theo BHXH Việt Nam, nhiều doanh nghiệp vẫn chỉ đóng BHXH cho người lao động theo mức lương.\
# Dù quy định từ 1/1/2018, tiền lương tháng đóng BHXH gồm mức lương và thêm khoản bổ sung khác.\
# BHXH Việt Nam vừa có báo cáo về tình hình thực hiện chính sách BHXH thời gian qua.\
# Theo đó, tình trạng nợ, trốn đóng BHXH, BHTN vẫn xảy ra ở hầu hết các tỉnh, thành.\
# Thống kê tới ngày 31/12/2020, tổng số nợ BHXH, BHYT, BHTN là hơn 13.500 tỷ đồng,\
# chiếm 3,35 % số phải thu, trong đó: Số nợ BHXH bắt buộc là hơn 8.600 tỷ đồng,\
# nợ BHTN là 335 tỷ đồng. Liên quan tới tiền lương đóng BHXH, báo cáo của\
# BHXH Việt Nam cho thấy: Nhiều doanh nghiệp vẫn chủ yếu xây dựng thang,\
# bảng lương để đóng BHXH bằng mức thấp nhất. Tức là bằng mức lương tối\
# thiểu vùng, cộng thêm 7 % đối với lao động đã qua đào tạo nghề và cộng\
# thêm 5 % hoặc 7 % đối với lao động làm nghề hoặc công việc nặng nhọc,\
# độc hại, nguy hiểm, đặc biệt nặng nhọc độc hại và nguy hiểm. Đối với\
# lao động giữ chức vụ, khoảng 80 % doanh nghiệp đã xây dựng thang,\
# bảng lương cụ thể theo chức danh. Đơn cử như với chức vụ giám đốc\
# sản xuất, giám đốc điều hành, trưởng phòng. Còn lại các doanh nghiệp\
# xây dựng đối với lao động giữ chức vụ theo thang lương, bảng lương\
# chuyên môn nghiệp vụ và bảng phụ cấp chức vụ, phụ cấp trách nhiệm.\
# Thống kê của BHXH Việt Nam cũng cho thấy, đa số doanh nghiệp đã đăng\
# ký đóng BHXH cho người lao động theo mức lương mà không có khoản bổ \
# sung khác. Mặc dù quy định từ ngày 1/1/2018, tiền lương tháng đóng BHXH \
# gồm mức lương và thêm khoản bổ sung khác."
# src = "Thông tư này hướng dẫn tuần tra, canh gác bảo vệ đê Điều trong mùa lũ đối với các tuyến đê sông được phân loại, phân cấp theo quy định tại Điều 4 của Luật Đê Điều."
# src = "Thông tư này hướng dẫn tuần tra. A B C."
# src = "1. Chấp hành sự phân công của Ban chỉ huy phòng, chống lụt, bão các cấp và chịu sự hướng dẫn về chuyên môn nghiệp vụ của cơ quan chuyên trách quản lý đê Điều.\n2. Tuần tra, canh gác và thường trực trên các điếm canh đê, khi có báo động lũ từ cấp I trở lên đối với tuyến sông có đê. Theo dõi diễn biến của đê Điều; phát hiện kịp thời những hư hỏng của đê Điều và báo cáo ngay cho Ban chỉ huy chống lụt bão xã, cán bộ chuyên trách quản lý đê Điều phụ trách tuyến đê đó và khẩn trương tiến hành xử lý giờ đầu theo đúng kỹ thuật đã được hướng dẫn.\n3. Tham gia xử lý sự cố và tu sửa kịp thời những hư hỏng của đê Điều, dưới sự hướng dẫn về kỹ thuật của cán bộ chuyên trách quản lý đê Điều hoặc ý kiến chỉ đạo của cấp trên.\n4. Canh gác, kiểm tra phát hiện và ngăn chặn kịp thời những hành vi vi phạm pháp Luật về đê Điều và phòng, chống lụt, bão và báo cáo ngay cán bộ chuyên trách quản lý đê Điều.\n5. Đeo phù hiệu khi làm nhiệm vụ."
# src = "Chấp hành sự phân công của Ban chỉ huy phòng, chống lụt, bão các cấp và chịu sự hướng dẫn về chuyên môn nghiệp vụ của cơ quan chuyên trách quản lý đê Điều. Tuần tra, canh gác và thường trực trên các điếm canh đê, khi có báo động lũ từ cấp I trở lên đối với tuyến sông có đê. Theo dõi diễn biến của đê Điều; phát hiện kịp thời những hư hỏng của đê Điều và báo cáo ngay cho Ban chỉ huy chống lụt bão xã, cán bộ chuyên trách quản lý đê Điều phụ trách tuyến đê đó và khẩn trương tiến hành xử lý giờ đầu theo đúng kỹ thuật đã được hướng dẫn. Tham gia xử lý sự cố và tu sửa kịp thời những hư hỏng của đê Điều, dưới sự hướng dẫn về kỹ thuật của cán bộ chuyên trách quản lý đê Điều hoặc ý kiến chỉ đạo của cấp trên. Canh gác, kiểm tra phát hiện và ngăn chặn kịp thời những hành vi vi phạm pháp Luật về đê Điều và phòng, chống lụt, bão và báo cáo ngay cán bộ chuyên trách quản lý đê Điều. Đeo phù hiệu khi làm nhiệm vụ."
# src = "Phạm vi tuần tra:\n Báo động lũ ở cấp I, bố trí người tuần tra như sau:\n- Lượt đi: 01 người (ban ngày), 02 người (ban đêm) kiểm tra mặt đê, mái đê phía sông, khu vực hành lang bảo vệ đê phía sông;\n- Lượt về: 01 người (ban ngày), 02 người (ban đêm) kiểm tra mái đê phía đồng, khu vực hành lang bảo vệ đê phía đồng, mặt ruộng, hồ ao gần chân đê phía đồng;\n Báo động lũ ở cấp II, bố trí người tuần tra như sau:\n- Lượt đi: 01 người kiểm tra mặt đê, mái đê phía sông, khu vực hành lang bảo vệ đê phía sông; 01 người (ban ngày), 02 người (ban đêm) kiểm tra mái đê phía đồng, khu vực hành lang bảo vệ đê phía đồng, mặt ruộng, hồ ao gần chân đê phía đồng;\n- Lượt về: 01 người (ban ngày), 02 người (ban đêm) kiểm tra mặt đê, mái đê phía sông, khu vực hành lang bảo vệ đê phía sông; 01 người kiểm tra mái đê phía đồng, khu vực hành lang bảo vệ đê phía đồng, mặt ruộng, hồ ao gần chân đê phía đồng;\n Báo động lũ ở cấp II và có tin bão khẩn cấp đổ bộ vào khu vực hoặc báo động lũ ở cấp III trở lên, bố trí người tuần tra như sau:\n- Lượt đi: 02 người kiểm tra mái đê, khu vực hành lang bảo vệ đê phía đồng, mặt ruộng, hồ ao gần chân đê phía đồng; 01 người kiểm tra mặt đê.\n- Lượt về: 02 người kiểm tra phía đồng; 01 người kiểm tra mặt đê, mái đê và khu vực hành lang bảo vệ đê phía sông.\n Mỗi kíp tuần tra phải kiểm tra vượt quá phạm vi phụ trách về hai phía, mỗi phía 50m. Đối với những khu vực đã từng xảy ra sự cố hư hỏng, phải kiểm tra quan sát rộng hơn để phát hiện sự cố.\n Người tuần tra, canh gác phải phát hiện kịp thời những hư hỏng của đê.\n Khi phát hiện có hư hỏng, người tuần tra phải tiến hành các công việc sau:\n Xác định loại hư hỏng, vị trí, đặc điểm, kích thước của loại hư hỏng;\n Xác định mực nước sông so với mặt đê tại vị trí phát sinh hư hỏng;\n Đánh dấu bằng cách ghi bảng, cắm tiêu báo hiệu vị trí hư hỏng; nếu sự cố nghiêm trọng, phải cấm người, vật, xe cơ giới đi qua và bố trí người canh gác tại chỗ để theo dõi thường xuyên diễn biến của hư hỏng;\n Báo cáo kịp thời và cụ thể tình hình hư hỏng cho đội trưởng hoặc đội phó, cán bộ chuyên trách quản lý đê Điều và Ban Chỉ huy phòng, chống lụt, bão xã."
src = "1. Phạm vi tuần tra:\na) Báo động lũ ở cấp I, bố trí người tuần tra như sau:\n- Lượt đi: 01 người (ban ngày), 02 người (ban đêm) kiểm tra mặt đê, mái đê phía sông, khu vực hành lang bảo vệ đê phía sông;\n- Lượt về: 01 người (ban ngày), 02 người (ban đêm) kiểm tra mái đê phía đồng, khu vực hành lang bảo vệ đê phía đồng, mặt ruộng, hồ ao gần chân đê phía đồng;\nb) Báo động lũ ở cấp II, bố trí người tuần tra như sau:\n- Lượt đi: 01 người kiểm tra mặt đê, mái đê phía sông, khu vực hành lang bảo vệ đê phía sông; 01 người (ban ngày), 02 người (ban đêm) kiểm tra mái đê phía đồng, khu vực hành lang bảo vệ đê phía đồng, mặt ruộng, hồ ao gần chân đê phía đồng;\n- Lượt về: 01 người (ban ngày), 02 người (ban đêm) kiểm tra mặt đê, mái đê phía sông, khu vực hành lang bảo vệ đê phía sông; 01 người kiểm tra mái đê phía đồng, khu vực hành lang bảo vệ đê phía đồng, mặt ruộng, hồ ao gần chân đê phía đồng;\nc) Báo động lũ ở cấp II và có tin bão khẩn cấp đổ bộ vào khu vực hoặc báo động lũ ở cấp III trở lên, bố trí người tuần tra như sau:\n- Lượt đi: 02 người kiểm tra mái đê, khu vực hành lang bảo vệ đê phía đồng, mặt ruộng, hồ ao gần chân đê phía đồng; 01 người kiểm tra mặt đê.\n- Lượt về: 02 người kiểm tra phía đồng; 01 người kiểm tra mặt đê, mái đê và khu vực hành lang bảo vệ đê phía sông.\nd) Mỗi kíp tuần tra phải kiểm tra vượt quá phạm vi phụ trách về hai phía, mỗi phía 50m. Đối với những khu vực đã từng xảy ra sự cố hư hỏng, phải kiểm tra quan sát rộng hơn để phát hiện sự cố.\n2. Người tuần tra, canh gác phải phát hiện kịp thời những hư hỏng của đê.\n3. Khi phát hiện có hư hỏng, người tuần tra phải tiến hành các công việc sau:\na) Xác định loại hư hỏng, vị trí, đặc điểm, kích thước của loại hư hỏng;\nb) Xác định mực nước sông so với mặt đê tại vị trí phát sinh hư hỏng;\nc) Đánh dấu bằng cách ghi bảng, cắm tiêu báo hiệu vị trí hư hỏng; nếu sự cố nghiêm trọng, phải cấm người, vật, xe cơ giới đi qua và bố trí người canh gác tại chỗ để theo dõi thường xuyên diễn biến của hư hỏng;\nd) Báo cáo kịp thời và cụ thể tình hình hư hỏng cho đội trưởng hoặc đội phó, cán bộ chuyên trách quản lý đê Điều và Ban Chỉ huy phòng, chống lụt, bão xã."
print(len(src))
tokenized_text = tokenizer.encode(src, return_tensors="pt").to(device)
# print(tokenized_text)
model.eval()
summary_ids = model.generate(
                    tokenized_text,
                    max_length=256, 
                    num_beams=5,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True
                )
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(output)
