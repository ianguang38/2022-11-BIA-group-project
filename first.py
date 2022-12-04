
from cnn_classification import predict
import sys
from PyQt5.QtGui import QPixmap, QColor, QBrush, QPainter
from PyQt5.QtWidgets import QWidget,QFileDialog,QGridLayout,QLabel,QApplication,QPushButton
from PyQt5.QtCore import Qt

class imgFrame(QWidget):
    def __init__(self,fn=None,v=None):      
        super().__init__()
        self.size=(256,256,20)
        self.img=fn
        self.value=v

        self.initUI()
        # self.show()

    def initUI(self):
        self.setMinimumSize(self.size[0],self.size[1]+self.size[2]+2)
        if self.img:
            fn=QLabel((self.img).split("/")[-1],self)
            fn.move(0,self.size[1])
        self.setAcceptDrops(True)

    
    def paintEvent(self, event):
        w=self.size[0]
        h=self.size[1]
        hs=self.size[2]
        qp = QPainter()
        qp.begin(self)

        if not self.img:
            qp.setBrush(QColor(255,255,255)) 
            qp.drawRect(0,0,w,h)
            qp.drawText(0,0,w,h, Qt.AlignCenter,"drag file here to import")
            qp.end()
            return
        

        qp.drawPixmap(0,0,QPixmap(self.img).scaled(w,h))
        qp.drawRect(0,h,w-1,hs)
        qp.setBrush(QColor(255, 175, 175)) 
        qp.drawRect(0,h,int(w*self.value), hs)
        qp.end()
    
    def change(self,d):
        self.img=d[0]
        self.value=d[1]
        self.repaint()

    
    def dropEvent(self, evn):
        fn=evn.mimeData().text()[8:]
        print(fn)
        self.change((fn,predict(fn)))
        evn.accept()

    def dragEnterEvent(self, evn):
        evn.accept()


class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(500, 300, 800, 500)
        self.setWindowTitle('lung cancer classification')
        self.initUI()
        self.show()

        self.data=[]

    def initUI(self):  

        rfb1=QPushButton("import image/images")
        rfb1.clicked.connect(self.importImg)

        rfb2=QPushButton("export")
        rfb2.clicked.connect(self.exportCsv)

        self.img=imgFrame()
        self.img.setAcceptDrops(True)

        
        layout=QGridLayout()
        layout.setSpacing(10)
        self.setLayout(layout)

        layout.addWidget(rfb1,1,0)
        layout.addWidget(rfb2,1,1)
        layout.addWidget(self.img,0,0)
        self.layout=layout
        



    def importImg(self):
        fns=QFileDialog.getOpenFileNames(self, "Open File", "./example img", "Image (*.jpg *.png *.tif)")
        for fn in fns[0]:
            pair=(fn,predict(fn))
            # self.img.change(pair)

            n=len(self.data)
            self.layout.addWidget(imgFrame(fn,pair[1]),n//5,n%5)

            self.data.append(pair)

    def exportCsv(self):
        save=QFileDialog.getSaveFileName(self, "Save as", "./example img", "csv (*.csv)")[0]
        with open(save,mode="w",encoding="utf-8") as f:
            for d in self.data:
                f.write(str(d[0])+","+str(d[1])+"\n")



        



if __name__ == '__main__':
  app = QApplication(sys.argv)
  demo =Example()
#   demo.show()
  sys.exit(app.exec_())