//
// Created by xinyan on 2020/2/29.
//

#pragma once

class ProgressBar {
 public:
  ProgressBar(int len, string message): len_(len), cur_(0), star_(0) {
    std::cout << "0%   10   20   30   40   50   60   70   80   90   100% \t " << message << std::endl
              << "|----|----|----|----|----|----|----|----|----|----| \t " << std::endl;
  }

  ProgressBar& update(int i) {
    cur_ += i;
    int num_star = 1.0 * cur_ / len_ * 50 + 1;
    if (num_star > star_) {
      for (int j = 0; j < num_star-star_; ++j) {
        std::cout << '*';
      }
      star_ = num_star;
      if (num_star==51) {
        std::cout << std::endl;
      }
    }

    return *this;
  }

  ProgressBar& operator++() {
    return update(1);
  }

  ProgressBar& operator+=(int i) {
    return update(i);
  }

 private:
  int len_;
  int cur_;
  int star_;
};
