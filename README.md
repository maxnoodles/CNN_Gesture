# CNN_Gesture 一个实时的手势识别脚本
基于Opencv+keras的实时手势识别系统， 准确率约96%  
  
python3.6 + opencv + keras + numpy + PIL  
  
运行"手势识别.py"，点击opencv的窗口, 按'l'进入手势录制模式，每录制一个训练手势会顺便录制测试集,测试集录制完5s后录制下一个手势  
全部训练手势录制完, 按't'进行训练，得到模型  
predict 可以查看每个手势的预测成功率  
打开 "手势识别.py" 中主函数的# models注释可实时预测手势，结果打印在opencv窗口中
