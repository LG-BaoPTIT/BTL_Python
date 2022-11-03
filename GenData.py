# GenData.py

#import sys
import sys
import numpy as np
import cv2


#Biến cấp độ modun
MIN_CONTOUR_AREA = 40


RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

############################
def main():
    imgTrainingNumbers = cv2.imread("training_chars.png")             #đọc trong hình ảnh số training
    #imgTrainingNumbers = cv2.resize(imgTrainingNumbers, dsize=(600,300))
    #imgTrainingNumbers = cv2.resize(imgTrainingNumbers, dsize = None, fx = 0.5, fy = 0.5)
    
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # lấy hình ảnh thang độ xám
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur
    # lọc hình ảnh từ thang độ xám sang đen trắng
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # làm cho các pixel vượt qua ngưỡng toàn màu trắng
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # sử dụng gaussian dường như cho kết quả tốt hơn
                                      cv2.THRESH_BINARY_INV,                # đảo ngược để nền trước sẽ có màu trắng, background sẽ là màu đen
                                      11,                                   # kích thước của vùng lân cận pixel được sử dụng để tính toán giá trị ngưỡng
                                      2)                                    # hằng số trừ đi giá trị trung bình hoặc giá trị trung bình có trọng số

    cv2.imshow("imgThresh", imgThresh)      # hiển thị hình ảnh ngưỡng để tham khảo

    imgThreshCopy = imgThresh.copy()        # tạo một bản sao của hình ảnh thresh, điều này trong b / c cần thiết findContours sửa đổi hình ảnh

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,        # hình ảnh đầu vào, hãy đảm bảo sử dụng một bản sao vì hàm sẽ sửa đổi hình ảnh này trong quá trình tìm kiếm đường viền
                                                 cv2.RETR_EXTERNAL,                 # chỉ lấy các đường viền ngoài cùng
                                                 cv2.CHAIN_APPROX_SIMPLE)           # nén các phân đoạn ngang, dọc và chéo và chỉ để lại các điểm cuối của chúng

                                # khai báo mảng numpy trống, chúng ta sẽ sử dụng nó để ghi vào tệp sau này
                                # không hàng, đủ cột để chứa tất cả dữ liệu hình ảnh
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
   

    intClassifications = []         # khai báo danh sách phân loại trống, đây sẽ là danh sách của chúng tôi về cách chúng tôi đang phân loại các ký tự của mình từ đầu vào của người dùng, chúng tôi sẽ ghi vào tệp ở cuối

                                    # các ký tự có thể mà chúng tôi quan tâm là các chữ số từ 0 đến 9, hãy đưa chúng vào danh sách intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')] #Là mã ascii của mấy chữ này

    for npaContour in npaContours:                          # for đường viền
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # nếu đường viền đủ lớn để xem xét
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # nhận và thoát ra khỏi trực tràng bị ràng buộc

                                                # vẽ hình chữ nhật xung quanh mỗi đường viền \
            cv2.rectangle(imgTrainingNumbers,           # vẽ hình chữ nhật trên hình ảnh đào tạo gốc
                          (intX, intY),                 # góc trên bên trái
                          (intX+intW,intY+intH),        # Góc dưới bên phải
                          (0, 0, 255),                  # red
                          2)                            # độ dày đường viền


            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # cắt ký tự ra khỏi ngưỡng hình ảnh
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # thay đổi kích thước hình ảnh, điều này sẽ nhất quán hơn để nhận dạng và lưu trữ

            cv2.imshow("imgROI", imgROI)                    # hiển thị biểu đồ đã cắt để tham khảo
            cv2.imshow("imgROIResized", imgROIResized)      # hiển thị hình ảnh đã thay đổi kích thước để tham khảo
            
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # hiển thị hình ảnh số đào tạo, bây giờ sẽ có các hình chữ nhật màu đỏ được vẽ trên đó

            intChar = cv2.waitKey(0)                     # get key press

            if intChar == 27:                   # nếu phím esc được nhấn
               sys.exit()
                                   # exit program
            elif intChar in intValidChars:      # nếu ký tự nằm trong danh sách ký tự mà chúng tôi đang tìm kiếm. . .

                intClassifications.append(intChar)        # nối ký tự phân loại vào danh sách ký tự số nguyên (chúng tôi sẽ chuyển đổi thành float sau trước khi ghi vào tệp)
                #Là file chứa label của tất cả các ảnh mẫu, tổng cộng có 32 x 5 = 160 mẫu.
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
                
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications, np.float32)                   # chuyển đổi danh sách phân loại của int thành mảng số nổi
    
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # san phẳng mảng nổi numpy thành 1d để chúng tôi có thể ghi vào tệp sau này

    print ("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           #ghi hình ảnh phẳng vào tệp
    np.savetxt("flattened_images.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # xóa các cửa sổ khỏi bộ nhớ
    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if




