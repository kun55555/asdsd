from flask import Flask, render_template, request, redirect, url_for
import pymysql

# 创建Flask应用程序
app = Flask(__name__)

# 设置数据库连接
connection = pymysql.connect(
    host='47.97.244.67',
    port=3306,
    user='root',
    password='20040227',
    db='test',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# 登录页面视图函数
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 获取登录表单的用户名和密码
        username = request.form['username']
        password = request.form['password']

        try:
            with connection.cursor() as cursor:
                # 查询数据库以验证用户凭据
                sql = "SELECT * FROM users WHERE username=%s AND password=%s "
                cursor.execute(sql, (username, password))
                user = cursor.fetchone()

                if user:
                    # 登录成功后执行的操作
                    return "登录成功"
                else:
                    # 登录失败时执行的操作
                    return "用户名或密码错误"
        except Exception as e:
            return "登录出错：" + str(e)
    return render_template('login.html')

# 注册页面视图函数
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # 获取注册表单的用户名和密码
        username = request.form['username']
        password = request.form['password']

        try:
            with connection.cursor() as cursor:
                # 检查用户名是否已经存在
                sql = "SELECT * FROM users WHERE username=%s"
                cursor.execute(sql, (username,))
                existing_user = cursor.fetchone()

                if existing_user:
                    return "用户名已存在，请选择另一个用户名"
                else:
                    # 将新用户插入数据库
                    insert_sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
                    cursor.execute(insert_sql, (username, password))
                    connection.commit()

                    return "注册成功"
        except Exception as e:
            return "注册出错：" + str(e)
    return render_template('register.html')

# 运行Flask应用程序
if __name__ == '__main__':
    app.run(debug=True)
