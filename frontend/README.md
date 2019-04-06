# 衣服推荐小程序设计

## 功能设计

1. 搭配广场--上传并得到推荐--
2. 个人中心--搭配管理--账号设置--个人信息--头像--积分
3. 



## 接口设计

### 登录

```javascript
url:user/userLogin
method:POST
content:
{
    code:(str)
}
return:
{
    state:#(后台设计状态码)
    userInfo:
    {
        userName:(str), # 用户名,规定用户名为唯一标识符并且不可改变
        userAvator:(str) # 头像，
        score: # 积分
    }
}
```

### 编辑信息

```javascript
url:user/userEdit
method:POST
content:
{
    userAvator: # 头像
    age: # 年龄
    sex: # 性别
    birth: # 出生年月
    phone: # 电话号码
    qq: # qq号
}
return:
{
    state:#(后台设计状态码)
}
```



### 广场列表

```javascript
url:square/indexList
method:POST
content:
{
    number:(int),# 需要返回的数量，-1表示全部
    random:(int)# 是否随机从数据库中返回 1表示随机，0表示不随机
}
return:
{
    state:,
    sqareList:
    [
        {
            clothId:(int), # 服装id（数据库唯一标识）
            name:(str), # 服装名
            type:(str),# 服装风格
            part:(str), # 服装部位 1 上装 2 夏装 3 头饰 4 鞋子 ...
            user:(str), # 所属用户
            goodsUrl,# 服装对应的购买链接
            picture:,# 服装照片
            description: # 衣服描述
            hot:, # 热度,点赞量
            comments:
    		[
                {
                	user: # 评论用户
                    content: # 评论内容
            	}
            ],
            
        }
    ]
}
```

### 点赞/取消点赞

```javascript
url:square/thumbs
method:POST
content:
{
    clothId:(int) # 服装id
    user: # 操作用户
    operation: #1 表示点赞 0 表示取消点赞
}
return:
{
    state:#(后台设计状态码)
}
```

### 评论/取消评论

```javascript
url:square/comment
method:POST
content:
{
    clothId:(int) # 服装id
    user: # 操作用户
    operation: #1 评论 0 表示删除评论
    comments : # 评论内容
}
return:
{
    state:#(后台设计状态码)
}
```

### 上传衣服/删除衣服

```javascript
url:square/addCloth
method:POST
content:
{
    user: # 操作用户
    operation: #1 上传衣服 0 表示删除衣服
    description： # 衣服描述
    clothPic: # 衣服照片,调用wx.uploadFile
    recommenedNum: # 推荐相关衣服数量
    clothType: # -1 表示在全部类型里推荐，其他值对应某种风格的衣服
    clothPart: # 推荐部位，如果用户选定了推荐部位的话
    clothId: # 要删除的衣服的唯一标识
}
return:
{
    state:#(后台设计状态码)
    clothId:# 数据库唯一标识符
    recommenedCloth:
    [
        {
            clothId: # 数据库唯一标识符
            name: # 衣服名称
            type: # 衣服类型
            part: # 衣服部位
            url: # 衣服购买链接
            picture: # 衣服照片
        }
    ]
}
```

### 搭配管理

```javascript
url:cloth/myCloth
method:GET
return:
{
    state:#(后台设计状态码),
    [
        {
            clothId:(int), # 服装id（数据库唯一标识）
            name:(str), # 服装名
            type:(str),# 服装风格
            part:(str), # 服装部位 1 上装 2 夏装 3 头饰 4 鞋子 ...
            goodsUrl,# 服装对应的购买链接
            picture:,# 服装照片
            hot:, # 热度,点赞量
            description: # 服装描述
        }
    ]
}
```







