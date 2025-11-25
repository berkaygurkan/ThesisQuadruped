# HÜCRE 3: Scientific Trainer (Augmented Support)

class ScientificTrainer:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger("ScientificTrainer")
        self.logger.setLevel(logging.INFO)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config bulunamadı: {config_path}")
            
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = f"{self.config['experiment']['name']}_{self.timestamp}"
        self.device = get_optimal_device()
        
        self.log_dir = os.path.join("..", "data", "logs", self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
            
        self.print_obsidian_header(config_path)

    def print_obsidian_header(self, original_path):
        print("\n" + "="*40)
        print("📋 OBSIDIAN LAB NOTE HEADER")
        print("="*40)
        print(f"run_id: \"{self.run_id}\"")
        print(f"source_config: \"{original_path}\"")
        print(f"device: \"{self.device}\"")
        print(f"input_type: \"Augmented (Obs+Act+Rew+Done)\"")
        print("="*40 + "\n")

    def _make_env_factory(self, seed, rank, wrapper_conf, force_cpu=False):
        def _init():
            env = gym.make(self.config['env']['id'])
            
            # 1. Mac Float32 Fix (En içte olmalı)
            device_type = "cpu" if force_cpu else str(self.device)
            if device_type == "mps": 
                env = Float32ObservationWrapper(env)
            
            # 2. Fizik Randomizasyonu
            env = RandomDampingWrapper(
                env, 
                min_damping=wrapper_conf['min_damping'], 
                max_damping=wrapper_conf['max_damping']
            )
            
            # 3. AUGMENTED OBS (Hocanın istediği yapı)
            # Bu wrapper, obs boyutunu büyütür.
            env = AugmentedObservationWrapper(env)
            
            env.reset(seed=seed + rank)
            return env
        return _init

    def train_variant(self, variant_name):
        print(f"\n{'='*60}")
        print(f"🚀 EĞİTİM BAŞLIYOR: {variant_name}")
        print(f"{'='*60}")
        
        seeds = self.config['training']['seeds']
        wrapper_conf = self.config['env']['wrappers'][0]['args']
        train_timesteps = self.config['training']['total_timesteps']
        
        trained_model_paths = []

        if variant_name == "LSTM":
            ModelClass = RecurrentPPO
            policy_type = "MlpLstmPolicy"
            current_device = self.device
            force_cpu = False
            policy_kwargs = self.config['hyperparameters']['policy_kwargs']
            print(f"   🧠 Model: RecurrentPPO (GPU/MPS)")
        else:
            ModelClass = PPO
            policy_type = "MlpPolicy"
            current_device = "cpu"
            force_cpu = True
            base_kwargs = self.config['hyperparameters']['policy_kwargs'].copy()
            exclude = ['lstm_hidden_size', 'n_lstm_layers', 'shared_lstm', 'enable_critic_lstm']
            policy_kwargs = {k: v for k, v in base_kwargs.items() if k not in exclude}
            print(f"   🧠 Model: Standard PPO (CPU Optimized)")

        for seed in seeds:
            set_random_seed(seed)
            
            env = SubprocVecEnv([
                self._make_env_factory(seed, i, wrapper_conf, force_cpu) 
                for i in range(self.config['env']['n_envs'])
            ])
            
            if variant_name == "FrameStack":
                env = VecFrameStack(env, n_stack=4)
                print(f"   🛠️ FrameStack: Aktif (4 frames)")
            
            env = VecMonitor(env, filename=os.path.join(self.log_dir, f"{variant_name}_seed_{seed}_monitor.csv"))
            
            model = ModelClass(
                policy=policy_type,
                env=env,
                verbose=1,
                device=current_device,
                tensorboard_log=self.log_dir,
                learning_rate=self.config['hyperparameters']['learning_rate'],
                n_steps=self.config['hyperparameters']['n_steps'],
                batch_size=self.config['hyperparameters']['batch_size'],
                gamma=self.config['hyperparameters']['gamma'],
                gae_lambda=self.config['hyperparameters']['gae_lambda'],
                ent_coef=self.config['hyperparameters']['ent_coef'],
                policy_kwargs=policy_kwargs
            )

            print(f"   🌱 Seed {seed} eğitiliyor... ({train_timesteps} steps)")
            
            model.learn(
                total_timesteps=train_timesteps, 
                tb_log_name=f"{variant_name}_seed_{seed}",
                progress_bar=False, 
                log_interval=1 # Sık loglama
            )
            
            save_path = os.path.join(self.log_dir, f"final_model_{variant_name}_seed_{seed}")
            model.save(save_path)
            trained_model_paths.append(save_path)
            print(f"   ✅ Model Kaydedildi: {save_path}")
            
            env.close()
            
        return trained_model_paths