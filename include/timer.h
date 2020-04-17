/** A timer object measures elapsed time,
 * and it is very similar to boost::timer. */
class timer {
 public:
  timer() { restart(); }
  ~timer() = default;
  /** Restart the timer. */
  void restart() {
    t_start = std::chrono::high_resolution_clock::now();
  }
  /** @return The elapsed time */
  double elapsed() {
    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  }

 private:
  std::chrono::high_resolution_clock::time_point t_start;
};
