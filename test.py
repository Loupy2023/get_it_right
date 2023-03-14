from predict import predict
from display import draw_img

# Test your functions before putting them on a api

test_image = "Path to my imge"  #.jpg format ?
prediction = predict(test_image)
to_print = draw_img(prediction)

print(to_print)
