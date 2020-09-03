import java.util.ArrayDeque;

public class CartPole {
    public static void main(String[] args) {

    }
}

class DQNAgent {
    private int state_size;
    private int action_size;
    private ArrayDeque<String> memory;
    private double alpha;
    private double gamma;
    private double exploration_rate;
    private double exploration_min;
    private double exploration_decay;
    private double learning_rate;
    private Sequential model;
    public DQNAgent(int observation_space, int action_space) {
        this.state_size = observation_space;
        this.action_size = action_space;
        //this.memory = deque(maxlen = 20000) // Тип collections.deque
        this.alpha = 1.0; //Скорость обучения агента
        this.gamma = 0.95;// Коэффициент уменьшения вознаграждения агента
        //Уровень обучения повышается с коэффициентом exploration_decay
        //Влияет на выбор действия action (0 или 1)
        this.exploration_rate = 1.0;
        this.exploration_min = 0.01;
        this.exploration_decay = 0.995;
        this.learning_rate = 0.001; //Скорость обучения сети
        this.model = this.build_model();
    }
    protected Sequential build_model() {
        Sequential model = null;
        return model;
    }
}
class Sequential {
    private int s;
    public Sequential(){
      s= Integer.parseInt(null);
    }
}