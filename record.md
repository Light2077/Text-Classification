2020/03/21:

我服了，每次新建一个项目都要重新查查步骤。怎么Git初始化，怎么生成gitignore，怎么把本地关联上Github。这也太鸡儿麻烦了。要想做一名合格的工程师熟练这些讲道理是必不可少的嗷。这个库是2020年1月12日建立的，之后就没怎么维护过了。而且，我，tm已经忘了这个库是怎么搭好的了。而且文件很混乱很烦。为了解决建库不熟练和库文件管理混乱这个痛点，我决定把这个库删了重新建立一下。

参考资料：

- [廖雪峰Git教程](https://www.liaoxuefeng.com/wiki/896043488029600)

### **1. 建库第一步**

打开gayhub，新建一个库，名字`Text-Classification`。勾上初始化README，添加python的`.gitignore`。

找一个风水宝地文件夹，在里边打开Git Bash，输入：

`git clone git@github.com:light2077/Text-Classification.git`

讲道理这个代码不难背啊，但是我老是忘记，好气啊。然后就获得了一个名为`Text-Classification`的文件夹，里边就是我的项目。

目前有：

```
.git
.gitignore
README.md
```

目前很完美，然后把这个文件`record.md`拷贝进`Text-Classification`文件夹。用git add, commit, push三连看看成功没。

擦，原来那么简单，为啥我以前就觉得那么麻烦呢？？？

### **2. pycharm关联github**

对着`Text-Classification`文件夹右键点击`open folder as pycharm project`。

`File->setting->Verison Control->Github`点击`Add account`添加github账号

然后`ctrl + k`commit一个文件试试。再然后`ctrl + shift + k`push到github

刷新github页面，看到了更新的内容，好了恭喜我。

### 3. 思考这个项目是做什么的

兄弟，文件夹之所以乱，就是因为不知道这个项目是来做什么的。首先我这个项目是文本分类项目。

开局一个`百度题库.zip`压缩包，~~代码全靠copy~~。

- 那得有一个文件件放数据吧。`data`
- 得有一个文件夹放一些辅助脚本，比如数据预处理。`utils`
- 得有三个文件夹放模型，因为我准备实现3个模型。`textcnn`, `transformer`, `bert`
- 得有一个文件夹放notebook，notebook这个东西是真的有毒。太好用了，以致于可能反过来会被这个工具拖住更进一步的脚步。`notebook`。额外说一点，为了`data`文件夹的独立性，readme里的图片统一存放在`notebook`文件夹里。

好了暂时就这么多。





## 其他烦人的点

**1. 如何在gitignore里忽略的文件夹下选择一个文件不忽略**：

```
/data/*
!/data/百度题库.zip
```

