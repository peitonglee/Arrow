#include <GL/glut.h>
#include <vector>

std::vector<std::vector<float>> lines; // 存储线段端点坐标

void init() {
  glClearColor(1.0, 1.0, 1.0, 0.0); // 设置清屏颜色为白色
  glColor3f(0.0, 0.0, 0.0);         // 设置绘制颜色为黑色
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, 1.0, 1.0, 100.0); // 设置投影矩阵
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); // 设置观察矩阵
}

void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清空颜色缓冲和深度缓冲
  glLoadIdentity();

  for (size_t i = 0; i < lines.size(); ++i) {
    glBegin(GL_LINES);
    glVertex3f(lines[i][0], lines[i][1], lines[i][2]); // 线段起点
    glVertex3f(lines[i][3], lines[i][4], lines[i][5]); // 线段终点
    glEnd();
  }

  glutSwapBuffers();
}

int main(int argc, char **argv) {
  // 初始化OpenGL
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(500, 500);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("3D Lines");

  init();

  // 添加示例线段数据
  lines.push_back({-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f}); // 示例线段1
  lines.push_back({1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f}); // 示例线段2

  glutDisplayFunc(display);
  glutMainLoop();

  return 0;
}
