# Awesome-LLM-Jailbreak

## Awesome LLM Jailbreak

Welcome to the **Awesome LLM Jailbreak** repository! This project curates a list of high-quality resources related to LLM Jailbreak, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-13

### 1. [Localizing Factual Inconsistencies in Attributable Text Generation](https://arxiv.org/pdf/2410.07473)

**Summary**: The paper introduces QASemConsistency, a method for precisely localizing factual inconsistencies in generated text by decomposing it into minimal predicate-argument propositions expressed as QA pairs and comparing them to a trusted reference. The approach is shown to be effective for both human annotation and automated detection using supervised models and large language models, achieving high inter-annotator agreement.

### 2. [AI-Press: A Multi-Agent News Generating and Feedback Simulation System Powered by Large Language Models](https://arxiv.org/pdf/2410.07561)

**Summary**: The paper introduces AI-Press, a multi-agent system leveraging large language models for automated news generation and refinement, addressing limitations in professionalism and ethical judgment. It also includes a feedback simulation system to predict public reactions based on demographic data, demonstrating enhanced news-generating capabilities and the effectiveness of feedback prediction through evaluations.

### 3. [StablePrompt: Automatic Prompt Tuning using Reinforcement Learning for Large Language Models](https://arxiv.org/pdf/2410.07652)

**Summary**: The paper introduces StablePrompt, a method for automatic prompt tuning in Large Language Models (LLMs) using Reinforcement Learning (RL). It addresses the instability of RL in prompt tuning by employing Adaptive Proximal Policy Optimization (APPO), which uses an LLM anchor model to adaptively adjust policy updates, ensuring stable and high-performance prompt generation across various tasks.

### 4. [Detecting Training Data of Large Language Models via Expectation Maximization](https://arxiv.org/pdf/2410.07582)

**Summary**: The paper introduces EM-MIA, a novel membership inference attack (MIA) method for LLMs that uses an expectation-maximization algorithm to iteratively refine membership and prefix scores, achieving state-of-the-art results on the WikiMIA dataset. Additionally, the authors present OLMoMIA, a benchmark for evaluating MIA methods, which allows for controlled assessment of MIA difficulty by varying the overlap between training and test data distributions.

### 5. [How Does Vision-Language Adaptation Impact the Safety of Vision Language Models?](https://arxiv.org/pdf/2410.07571)

**Summary**: The study investigates how Vision-Language adaptation (VL adaptation) impacts the safety of Large Vision-Language Models (LVLMs), finding that safety is compromised during the transformation from Large Language Models (LLMs). Although safety fine-tuning methods can mitigate some risks, they often lead to reduced helpfulness due to over-rejection. The paper proposes weight merging as a solution to balance safety and helpfulness in LVLMs.

### 6. [MACPO: Weak-to-Strong Alignment via Multi-Agent Contrastive Preference Optimization](https://arxiv.org/pdf/2410.07672)

**Summary**: The paper introduces MACPO, a multi-agent contrastive preference optimization framework designed to address the weak-to-strong alignment problem in LLMs. MACPO enables weak teachers and strong students to iteratively learn from each other by reinforcing positive behaviors and penalizing negative ones, using strategies like mutual positive behavior augmentation and hard negative behavior construction. Experimental results show that MACPO improves alignment performance for both strong students and weak teachers, with enhanced performance as the number of weak teachers increases.

### 7. [Uncovering Overfitting in Large Language Model Editing](https://arxiv.org/pdf/2410.07819)

**Summary**: The paper identifies a phenomenon called Editing Overfit in Large Language Models, where edited models overly prioritize the edit target, leading to poor generalization in complex tasks. To address this, the authors introduce a new benchmark, EVOKE, and propose a strategy called Learn to Inference (LTI) with a Multi-stage Inference Constraint module, which helps edited models recall knowledge more effectively, reducing overfitting.

### 8. [Fine-Tuning Language Models for Ethical Ambiguity: A Comparative Study of Alignment with Human Responses](https://arxiv.org/pdf/2410.07826)

**Summary**: The study investigates the alignment of language models with human judgments in morally ambiguous scenarios by fine-tuning models on curated datasets from the Scruples project. Fine-tuning significantly improved model performance, particularly in cross-entropy and Dirichlet scores, with Mistral-7B-Instruct-v0.3 achieving results comparable to GPT-4. However, BERT and RoBERTa models still outperformed the experimental models in cross-entropy scores, highlighting the need for further research to enhance ethical reasoning in language models.

### 9. [Private Language Models via Truncated Laplacian Mechanism](https://arxiv.org/pdf/2410.08027)

**Summary**: The paper introduces a novel private embedding method called the high dimensional truncated Laplacian mechanism, which extends the truncated Laplacian mechanism to higher dimensions to improve privacy protection in NLP tasks. The authors demonstrate that their method has lower variance compared to existing techniques and maintains high utility even in high privacy regimes, as evidenced by experiments on three datasets.

### 10. [The Rise of AI-Generated Content in Wikipedia](https://arxiv.org/pdf/2410.08044)

**Summary**: The paper investigates the increasing presence of AI-generated content on Wikipedia, highlighting concerns about accuracy, bias, and accountability. Using AI detectors, the study finds that over 5% of newly created English Wikipedia articles are flagged as AI-generated, with lower percentages in other languages, and these flagged articles often exhibit lower quality and biased content.

### 11. [COMPL-AI Framework: A Technical Interpretation and LLM Benchmarking Suite for the EU Artificial Intelligence Act](https://arxiv.org/pdf/2410.07959)

**Summary**: The paper introduces COMPL-AI, a framework that provides a technical interpretation of the EU's Artificial Intelligence Act (AI Act) and an open-source benchmarking suite for LLMs. It evaluates 12 prominent LLMs, revealing deficiencies in robustness, safety, diversity, and fairness, and underscores the need for more balanced LLM development and regulation-aligned benchmarks. This work demonstrates the challenges and potential of translating the AI Act's obligations into actionable technical requirements.

### 12. [A Target-Aware Analysis of Data Augmentation for Hate Speech Detection](https://arxiv.org/pdf/2410.08053)

**Summary**: The paper explores the use of data augmentation with generative language models to improve hate speech detection, particularly for underrepresented identity groups. By augmenting a dataset with synthetic examples, the study finds that combining traditional data augmentation methods with generative models yields the best results, significantly enhancing classification performance for specific hate categories. This approach aims to create more inclusive and effective hate speech detection systems.

### 13. [Human and LLM Biases in Hate Speech Annotations: A Socio-Demographic Analysis of Annotators and Targets](https://arxiv.org/pdf/2410.07991)

**Summary**: The paper investigates how the socio-demographic characteristics of annotators and targets influence biases in hate speech annotations, revealing significant differences in bias intensity and prevalence. It also compares these human biases with those exhibited by persona-based LLMs, finding that while LLMs do show biases, they differ substantially from those of human annotators. The study provides insights into mitigating biases in AI-driven hate speech detection systems.

### 14. [A Closer Look at Machine Unlearning for Large Language Models](https://arxiv.org/pdf/2410.08109)

**Summary**: The paper examines the challenges of machine unlearning in LLMs, particularly in removing specific content without compromising overall performance. It introduces new evaluation metrics and proposes methods like maximizing entropy for untargeted unlearning and answer preservation loss for targeted unlearning to address existing issues. Experimental results show the effectiveness of these approaches across various unlearning scenarios.

### 15. [Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-ended Question Answering](https://arxiv.org/pdf/2410.08085)

**Summary**: The paper introduces OKGQA, a new benchmark for evaluating the effectiveness of Knowledge Graphs (KGs) in enhancing Large Language Models (LLMs) in open-ended question answering, aiming to reduce hallucinations and improve reasoning. Additionally, the study introduces OKGQA-P to assess model performance under perturbed KG conditions, providing insights into the robustness and trustworthiness of LLMs when integrated with KGs.

### 16. [Robust AI-Generated Text Detection by Restricted Embeddings](https://arxiv.org/pdf/2410.08113)

**Summary**: The paper introduces a method for robust AI-generated text detection by focusing on the geometry of embedding spaces in Transformer models. By removing harmful linear subspaces, the approach enhances the classifier's ability to generalize across different generators and domains, achieving significant improvements in out-of-distribution classification scores. The authors release their code and data to support further research.

### 17. [GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-time Alignment](https://arxiv.org/pdf/2410.08193)

**Summary**: The paper introduces GenARM, a test-time alignment method that uses an Autoregressive Reward Model to guide LLMs during autoregressive text generation without retraining. GenARM outperforms existing test-time alignment methods and matches the performance of training-time methods, while also enabling efficient multi-objective alignment and catering to diverse user preferences.

### 18. [Insight Over Sight? Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](https://arxiv.org/pdf/2410.08145)

**Summary**: The paper investigates the issue of vision-knowledge conflicts in Multimodal Large Language Models (MLLMs), where visual information contradicts the model's internal commonsense knowledge. It introduces a benchmark with 374 images and 1,122 QA pairs to assess conflict resolution, finding that models often over-rely on textual data. A new prompting strategy, "Focus-on-Vision" (FoV), is proposed to improve models' reliance on visual data, enhancing their conflict-resolution capabilities.

### 19. [Sample then Identify: A General Framework for Risk Control and Assessment in Multimodal Large Language Models](https://arxiv.org/pdf/2410.08174)

**Summary**: The paper introduces TRON, a novel framework for risk control and assessment in Multimodal Large Language Models (MLLMs), addressing the limitations of existing methods that rely on internal model logits or are restricted to multiple-choice settings. TRON uses a two-step process involving conformal scores for sampling response sets and nonconformity scores for identifying high-quality responses, ensuring adaptability in both open-ended and closed-ended scenarios. The framework demonstrates effective risk control across various datasets, maintaining efficiency and stability in risk assessment.

### 20. [The First VoicePrivacy Attacker Challenge Evaluation Plan](https://arxiv.org/pdf/2410.07428)

**Summary**: The paper introduces the First VoicePrivacy Attacker Challenge, part of the VoicePrivacy initiative, aimed at developing systems to attack voice anonymization methods. Participants will create automatic speaker verification systems to evaluate anonymization techniques from the VoicePrivacy 2024 Challenge, with evaluation based on equal error rate (EER). The top performers will present their results at ICASSP 2025.

### 21. [No Free Lunch: Retrieval-Augmented Generation Undermines Fairness in LLMs, Even for Vigilant Users](https://arxiv.org/pdf/2410.07589)

**Summary**: The paper investigates the fairness implications of Retrieval-Augmented Generation (RAG) in LLMs, finding that even with fully censored datasets, RAG can produce biased outputs without requiring fine-tuning or retraining. The study highlights the need for new strategies to ensure fairness in RAG-based LLMs and proposes potential mitigations for further research.

### 22. [SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection](https://arxiv.org/pdf/2410.07471)

**Summary**: The paper introduces SEAL, a framework for enhancing the safety and alignment of Large Language Models (LLMs) during fine-tuning by using bilevel optimization to rank data. SEAL prioritizes safe and high-quality data while downranking unsafe or low-quality data, resulting in models that outperform baselines by significant margins, as demonstrated on Llama-3-8b-Instruct and Merlinite-7b models.

### 23. [Mitigating Gender Bias in Code Large Language Models via Model Editing](https://arxiv.org/pdf/2410.07820)

**Summary**: The paper introduces CodeGenBias, a dataset and FB-Score metric to evaluate gender bias in code Large Language Models (LLMs), and proposes MG-Editing, a multi-granularity model editing approach to mitigate this bias. Experiments show that MG-Editing effectively reduces gender bias while preserving code generation capabilities, with the best performance at row and neuron levels of granularity.

### 24. [Learn from Real: Reality Defender's Submission to ASVspoof5 Challenge](https://arxiv.org/pdf/2410.07379)

**Summary**: The paper introduces Reality Defender's submission to the ASVspoof5 challenge, featuring a novel pretraining strategy that enhances the generalizability of audio deepfake detection models while keeping computational costs low. The system, SLIM, uses self-supervised contrastive learning to extract style-linguistics dependency embeddings from genuine speech, aiding in distinguishing between spoofed and bonafide audio. The submission achieved strong performance metrics across multiple datasets, including a minDCF of 0.1499 and EER of 5.5% on ASVspoof5 Track 1.

### 25. [Evolutionary Contrastive Distillation for Language Model Alignment](https://arxiv.org/pdf/2410.07513)

**Summary**: The paper introduces Evolutionary Contrastive Distillation (ECD), a method to enhance language models' ability to follow complex instructions by generating synthetic preference data that highlights the differences between successful and subtly flawed responses. ECD achieves this by progressively evolving simple instructions into more complex ones, creating "hard negative" responses that nearly meet the new requirements but miss a few key points. The method is shown to improve instruction-following performance, making a 7B model competitive with 70B models.

### 26. [Automatic Curriculum Expert Iteration for Reliable LLM Reasoning](https://arxiv.org/pdf/2410.07627)

**Summary**: The paper introduces Automatic Curriculum Expert Iteration (Auto-CEI) to address hallucinations and laziness in Large Language Model (LLM) reasoning by enhancing the model's ability to assertively answer within its capabilities and decline when tasks exceed them. Auto-CEI uses Expert Iteration to guide reasoning trajectories and reduce errors, while a curriculum adjusts rewards to encourage extended reasoning before acknowledging incapability, resulting in superior alignment and balance between assertiveness and conservativeness across various tasks.

### 27. [Fuse to Forget: Bias Reduction and Selective Memorization through Model Fusion](https://arxiv.org/pdf/2311.07682)

**Summary**: The paper explores the use of model fusion to reduce unwanted knowledge, such as shortcuts, social biases, and memorization of training data, in fine-tuned language models. Through experiments, it shows that model fusion enhances shared knowledge while forgetting unshared knowledge, suggesting its potential as a debiasing tool and a method to address privacy concerns in language models.

### 28. [Steering Language Models With Activation Engineering](https://arxiv.org/pdf/2308.10248)

**Summary**: The paper introduces Activation Engineering, specifically the Activation Addition (ActAdd) technique, which modifies intermediate activations during inference to steer model outputs. ActAdd uses contrasting prompt pairs to compute steering vectors, achieving state-of-the-art results in sentiment shift and detoxification while maintaining performance on unrelated tasks. This method is lightweight, requiring no machine optimization and minimal data, enabling rapid iteration and inference-time control over output properties.

### 29. [LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation](https://arxiv.org/pdf/2401.17244)

**Summary**: The paper introduces LLaMP, a multimodal retrieval-augmented generation framework that enhances Large Language Models (LLMs) for high-fidelity materials knowledge retrieval and distillation. LLaMP employs hierarchical reasoning-and-acting agents to interact dynamically with materials data and simulations, significantly reducing hallucinations and improving accuracy in materials science tasks. The framework demonstrates strong tool usage and self-consistency, offering a nearly hallucination-free approach to materials informatics.

### 30. [DiaHalu: A Dialogue-level Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/pdf/2403.00896)

**Summary**: The paper introduces DiaHalu, the first dialogue-level hallucination evaluation benchmark for LLMs, addressing the limitations of existing benchmarks that focus on sentence or passage levels and often ignore faithfulness hallucination. DiaHalu simulates authentic human-machine interactions by having ChatGPT3.5 generate dialogues on various topics, which are then manually modified and annotated by scholars, covering multiple dialogue domains and hallucination subtypes.

### 31. [How Likely Do LLMs with CoT Mimic Human Reasoning?](https://arxiv.org/pdf/2402.16048)

**Summary**: The paper investigates the reasoning processes of Large Language Models (LLMs) using Chain-of-Thought (CoT) by comparing them to human reasoning through causal analysis. It finds that LLMs often exhibit spurious correlations and consistency errors, and that in-context learning strengthens causal structure while post-training techniques weaken it. The study suggests that model size alone does not improve causal reasoning, highlighting the need for new techniques to enhance LLM reasoning capabilities.

### 32. [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/pdf/2404.10719)

**Summary**: The paper investigates the performance of Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO) in aligning LLMs with human preferences, finding that PPO outperforms DPO across various benchmarks, including challenging code competitions. The study reveals key factors contributing to PPO's superior performance and suggests that DPO has fundamental limitations.

### 33. [Protecting Your LLMs with Information Bottleneck](https://arxiv.org/pdf/2404.13968)

**Summary**: The paper introduces the Information Bottleneck Protector (IBProtector), a defense mechanism for LLMs that aims to mitigate jailbreak attacks by selectively compressing and perturbing prompts. IBProtector uses a lightweight extractor to preserve essential information while avoiding trivial solutions, and it can operate without access to gradients, making it compatible with any LLM. Empirical evaluations demonstrate that IBProtector effectively reduces jailbreak attempts while maintaining response quality and inference speed, outperforming existing defense methods.

### 34. [Self-Recognition in Language Models](https://arxiv.org/pdf/2407.06946)

**Summary**: The paper investigates the potential for self-recognition in language models (LMs) by proposing a novel test using model-generated "security questions." The study, conducted on ten prominent LMs, found no evidence of self-recognition but revealed that LMs tend to select the "best" answer regardless of its origin, with consistent preferences across models. The research also highlights position bias in multiple-choice settings.

### 35. [VIVA: A Benchmark for Vision-Grounded Decision-Making with Human Values](https://arxiv.org/pdf/2407.03000)

**Summary**: The paper introduces VIVA, a benchmark designed to evaluate the ability of vision-language models (VLMs) to make decisions grounded in human values based on visual scenarios. VIVA includes 1,240 images with annotated decisions, challenging models to select appropriate actions and justify them with relevant human values. Experiments reveal limitations in current VLMs' ability to integrate human values in decision-making, suggesting the need for improvements in understanding action consequences and predicting human values.

### 36. [Edu-Values: Towards Evaluating the Chinese Education Values of Large Language Models](https://arxiv.org/pdf/2409.12739)

**Summary**: The paper introduces Edu-Values, a benchmark for evaluating the alignment of LLMs with Chinese educational values across seven dimensions. The study finds that Chinese LLMs outperform English ones, with Qwen 2 scoring highest, and highlights areas where LLMs excel (subject knowledge, teaching skills) and struggle (professional ethics, basic competencies). The benchmark includes diverse question types and is available for further research.

### 37. [Who's in and who's out? A case study of multimodal CLIP-filtering in DataComp](https://arxiv.org/pdf/2405.08209)

**Summary**: The paper examines the biases in image-text data filtering using CLIP on the DataComp dataset, finding that marginalized groups like LGBTQ+ individuals and older women are disproportionately excluded. It also reveals that the NSFW filter is ineffective and that copyrighted content is frequently included, highlighting the need for improved dataset creation and filtering practices.

### 38. [How Does Diverse Interpretability of Textual Prompts Impact Medical Vision-Language Zero-Shot Tasks?](https://arxiv.org/pdf/2409.00543)

**Summary**: The paper investigates the impact of diverse textual prompts on the performance of medical vision-language pre-training (MedVLP) models in zero-shot tasks, finding that these models exhibit unstable performance across various prompt styles, particularly with more complex medical concepts. The study highlights the need for improved robustness in MedVLP methodologies to handle diverse prompts effectively.

### 39. [SciSafeEval: A Comprehensive Benchmark for Safety Alignment of Large Language Models in Scientific Tasks](https://arxiv.org/pdf/2410.03769)

**Summary**: The paper introduces SciSafeEval, a comprehensive benchmark for evaluating the safety alignment of large language models (LLMs) in scientific tasks across various domains and representations, including molecular, protein, and genomic languages. The benchmark includes zero-shot, few-shot, and chain-of-thought evaluations, along with a 'jailbreak' feature to test LLMs' defenses against malicious intent, aiming to promote responsible development and deployment of LLMs in scientific research.

### 40. [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs](https://arxiv.org/pdf/2410.03768)

**Summary**: The paper explores the emergence of steganographic collusion in large language models (LLMs) and demonstrates that such collusion can arise indirectly from optimization pressures. The authors introduce two methods, gradient-based and in-context reinforcement learning, to elicit sophisticated steganographic communication in LLMs, finding that it can be robust to both passive and active mitigation efforts. The study emphasizes the need for innovative oversight techniques to effectively mitigate risks post-deployment.

### 41. [You Know What I'm Saying -- Jailbreak Attack via Implicit Reference](https://arxiv.org/pdf/2410.03857)

**Summary**: The paper introduces a novel vulnerability called Attack via Implicit Reference (AIR), which exploits context within nested harmless objectives to generate malicious content undetected by current large language models (LLMs). AIR achieves over 90% attack success rates across various state-of-the-art models, including GPT-4, Claude-3.5, and Qwen-2-72B, highlighting the need for improved defense mechanisms against contextual attacks.

### 42. [Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](https://arxiv.org/pdf/2410.03869)

**Summary**: The paper introduces a novel jailbreaking method called Chain-of-Jailbreak (CoJ) attack, which exploits text-based image generation models by decomposing malicious queries into multiple sub-queries for iterative image editing, bypassing safety measures. The CoJ attack was found to be effective in over 60% of cases across various models, outperforming other methods. To counter this, the authors propose a defense mechanism called Think Twice Prompting, which successfully defends against 95% of CoJ attacks.

### 43. [Towards Secure Tuning: Mitigating Security Risks Arising from Benign Instruction Fine-Tuning](https://arxiv.org/pdf/2410.04524)

**Summary**: The paper addresses security risks in Large Language Models (LLMs) after Instruction Fine-Tuning (IFT), even when the tuning instructions are benign. The authors propose a novel IFT strategy called Modular Layer-wise Learning Rate (ML-LR), which differentiates learning rates for robust modules identified through a proxy-guided search algorithm. Experimental results show that this strategy effectively mitigates security risks without compromising the usability or expertise of the LLMs.

### 44. [A test suite of prompt injection attacks for LLM-based machine translation](https://arxiv.org/pdf/2410.05047)

**Summary**: The paper introduces a comprehensive test suite for evaluating prompt injection attacks (PIAs) on LLM-based machine translation systems, extending previous work by Sun and Miceli-Barone. The suite covers all language pairs in the WMT 2024 General Machine Translation task and includes various attack formats to assess the robustness of these systems against malicious input interference.

### 45. [Harnessing Task Overload for Scalable Jailbreak Attacks on Large Language Models](https://arxiv.org/pdf/2410.04190)

**Summary**: The paper introduces a scalable jailbreak attack on Large Language Models (LLMs) that exploits resource constraints to bypass safety mechanisms. By engaging the LLM in a computationally intensive preliminary task, the attack saturates the model's processing capacity, preventing the activation of safety protocols when executing the target instruction. This method demonstrates high success rates across various LLMs and emphasizes the need for more robust safety measures that consider resource limitations.

### 46. [Suspiciousness of Adversarial Texts to Human](https://arxiv.org/pdf/2410.04377)

**Summary**: The paper investigates the concept of human suspiciousness in adversarial texts, which differs from imperceptibility in images as texts must maintain semantic coherence while remaining undetected by human readers. The study introduces a novel dataset of human evaluations on the suspiciousness of adversarial sentences and develops a regression model to quantify and reduce this suspiciousness, providing a baseline for future research in adversarial text generation.

### 47. [Prompts have evil twins](https://arxiv.org/pdf/2311.07064)

**Summary**: The paper introduces "evil twins," unintelligible prompts that elicit similar behavior in language models as their natural-language counterparts, despite being uninterpretable to humans. These prompts are shown to transfer across different models and are generated by solving a maximum-likelihood problem, which has broader applications in understanding and manipulating model behavior.

### 48. [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)

**Summary**: The paper investigates the vulnerability of Retrieval-Augmented Generation (RAG) systems, particularly those using instruction-tuned Language Models (LMs), to datastore leakage through prompt injection. It demonstrates that adversaries can exploit these systems to extract verbatim text data, with the risk increasing as model size scales up. The study also shows that position bias elimination strategies can mitigate this vulnerability.

### 49. [Representation noising effectively prevents harmful fine-tuning on LLMs](https://arxiv.org/pdf/2405.14577)

**Summary**: The paper introduces Representation Noising (RepNoise), a defense mechanism against harmful fine-tuning attacks on large language models (LLMs). RepNoise removes information related to harmful representations, making it difficult for attackers to recover them during fine-tuning, even when they have access to the model weights. The method is shown to be effective across various harmful tasks and does not impair the model's performance on harmless tasks.

### 50. [Evaluating and Safeguarding the Adversarial Robustness of Retrieval-Based In-Context Learning](https://arxiv.org/pdf/2405.15984)

**Summary**: The paper investigates the robustness of Retrieval-Augmented In-Context Learning (ICL) methods against adversarial attacks, finding that while they improve robustness against test sample attacks, they are more vulnerable to demonstration attacks. The study introduces a training-free defense method, DARD, which enhances robustness by enriching the example pool with attacked samples, achieving a 15% reduction in Attack Success Rate (ASR) compared to baselines.

### 51. [Understanding Jailbreak Success: A Study of Latent Space Dynamics in Large Language Models](https://arxiv.org/pdf/2406.09289)

**Summary**: The paper investigates the mechanisms behind jailbreaking in large language models, finding that a single jailbreak vector can mitigate different types of jailbreaks, suggesting a common internal mechanism. It also identifies a potential commonality in how effective jailbreaks reduce the model's perception of prompt harmfulness, providing insights for developing more robust countermeasures against jailbreaking.

### 52. [Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective](https://arxiv.org/pdf/2407.16997)

**Summary**: The paper revisits the Who's Harry Potter (WHP) method for Large Language Model (LLM) unlearning, proposing a new task of targeted unlearning where only specific information about a given target is removed from the model. It introduces a causal intervention framework to model the unlearning process, treating the target's knowledge as a confounder and the unlearning as a deconfounding process. The proposed approach demonstrates competitive performance in experiments without explicit optimization for specific criteria.

### 53. [Advancing Adversarial Suffix Transfer Learning on Aligned Large Language Models](https://arxiv.org/pdf/2408.14866)

**Summary**: The paper introduces DeGCG, a two-stage transfer learning framework for improving the efficiency of adversarial suffix generation in large language models (LLMs). By decoupling the search process into pre-searching and post-searching stages, DeGCG enhances suffix transferability across models and datasets. The interleaved variant, i-DeGCG, further accelerates the search process by leveraging self-transferability, achieving significant improvements in adversarial success rates (ASRs) on Llama2-chat-7b.

### 54. [Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm](https://arxiv.org/pdf/2409.14119)

**Summary**: The paper introduces Obliviate, a defense mechanism for neutralizing task-agnostic backdoors in parameter-efficient fine-tuning (PEFT) of large language models. The proposed method, which amplifies benign neurons and penalizes trigger tokens, significantly reduces the success rate of state-of-the-art backdoor attacks by 83.6% across various PEFT architectures. Obliviate also demonstrates robust defense against both task-specific backdoors and adaptive attacks.

### 55. [Reconstruct Your Previous Conversations! Comprehensively Investigating Privacy Leakage Risks in Conversations with GPT Models](https://arxiv.org/pdf/2402.02987)

**Summary**: The paper introduces a Conversation Reconstruction Attack that aims to extract private conversation content between users and GPT models through malicious prompts. Despite GPT-4's resilience, advanced attacks show significant privacy leakage across all models. The study underscores the vulnerability of GPT models to privacy breaches and calls for stronger safeguards.

### 56. [An Adversarial Perspective on Machine Unlearning for AI Safety](https://arxiv.org/pdf/2409.18025)

**Summary**: The paper examines the effectiveness of unlearning methods in removing hazardous capabilities from large language models, challenging the distinction between unlearning and traditional safety post-training. It demonstrates that existing jailbreak techniques can bypass unlearning protections when applied strategically and introduces adaptive methods that recover most unlearned capabilities, questioning the robustness of current unlearning approaches.

### 57. [Adversarial Suffixes May Be Features Too!](https://arxiv.org/pdf/2410.00451)

**Summary**: The paper investigates the hypothesis that adversarial suffixes in large language models (LLMs) are not just vulnerabilities but may represent features that can dominate model behavior. Through experiments, the authors demonstrate that benign features can be transformed into adversarial suffixes that compromise safety alignment and that these features can be introduced through fine-tuning with benign datasets. The findings underscore the risk posed by benign features in training data and advocate for further research to enhance LLM safety.

### 58. [Automated Progressive Red Teaming](https://arxiv.org/pdf/2407.03876)

**Summary**: The paper introduces Automated Progressive Red Teaming (APRT), a framework designed to identify vulnerabilities in large language models (LLMs) by automating the process of generating adversarial prompts. APRT uses three core modules—Intention Expanding LLM, Intention Hiding LLM, and Evil Maker—to progressively explore and exploit LLM weaknesses through multi-round interactions. The framework's effectiveness is demonstrated through extensive experiments, showing it can elicit unsafe but useful responses from various LLMs, including Meta's Llama-3-8B-Instruct, GPT-4o, and Claude-3.5.

### 59. [Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models](https://arxiv.org/pdf/2410.02298)

**Summary**: The paper introduces Jailbreak Antidote, a method for dynamically adjusting the safety-utility balance in large language models (LLMs) by manipulating a sparse subset of the model's internal states during inference. This approach allows for real-time control over safety preferences without increasing computational overhead or inference latency, and it is shown to be effective across a range of LLMs and against various jailbreak attacks.



---

*Last updated on 2024-10-13*