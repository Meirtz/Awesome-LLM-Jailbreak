# Awesome-LLM-Jailbreak

## Awesome LLM Jailbreak

Welcome to the **Awesome LLM Jailbreak** repository! This project curates a list of high-quality resources related to LLM Jailbreak, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-18

### 1. [UniAutoML: A Human-Centered Framework for Unified Discriminative and Generative AutoML with Large Language Models](https://arxiv.org/pdf/2410.12841)

**Summary**: The paper introduces UniAutoML, a human-centered AutoML framework that integrates Large Language Models (LLMs) to support both discriminative and generative tasks. It features a conversational user interface for real-time guidance and feedback, enhancing transparency and user control, and has been validated through experiments and user studies to improve performance and trust in AutoML processes.

### 2. [A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions](https://arxiv.org/pdf/2410.12837)

**Summary**: The paper provides a thorough overview of Retrieval-Augmented Generation (RAG), detailing its evolution, current advancements, and future directions. It highlights how RAG integrates retrieval mechanisms with generative models to improve accuracy in knowledge-intensive tasks, while also addressing challenges like scalability and ethical concerns. The survey aims to guide researchers and practitioners in leveraging RAG's potential in natural language processing.

### 3. [Capturing Bias Diversity in LLMs](https://arxiv.org/pdf/2410.12839)

**Summary**: The paper introduces BiasGPT, a framework that enhances Large Language Models (LLMs) by incorporating diverse biases reflecting demographic characteristics like gender, age, and race. By customizing multiple GPT instances, each with distinct biases, the study demonstrates how these models can collaboratively generate more inclusive and representative AI responses, capturing a broader range of human experiences and viewpoints.

### 4. [Prompt Engineering a Schizophrenia Chatbot: Utilizing a Multi-Agent Approach for Enhanced Compliance with Prompt Instructions](https://arxiv.org/pdf/2410.12848)

**Summary**: The paper introduces a multi-agent approach to improve the compliance of a schizophrenia chatbot with prompt instructions, addressing concerns about the ethical and safety implications of Large Language Models (LLMs). By employing a Critical Analysis Filter, which involves a team of LLM agents analyzing and refining the chatbot's responses, the study demonstrates a significant increase in compliance scores, suggesting that this method can enhance the safe and effective use of LLMs in mental health education platforms.

### 5. [JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework](https://arxiv.org/pdf/2410.12855)

**Summary**: The paper introduces JAILJUDGE, a comprehensive benchmark for evaluating Large Language Model (LLM) defenses against jailbreak attacks, featuring diverse risk scenarios and multilingual prompts. It includes the JailJudge MultiAgent framework for explainable, fine-grained scoring and JAILJUDGE Guard, an end-to-end judge model that provides reasoning. The study demonstrates state-of-the-art performance in jailbreak attack and defense tasks, with JailBoost and GuardShield significantly enhancing attack and defense effectiveness.

### 6. [Improving Instruction-Following in Language Models through Activation Steering](https://arxiv.org/pdf/2410.12877)

**Summary**: The paper introduces a method called Activation Steering, which uses instruction-specific vector representations derived from language models to enhance their ability to follow instructions. By computing the difference in activations between inputs with and without instructions, the method allows for modular control over model outputs, improving adherence to constraints such as format, length, and word inclusion. The approach is shown to be effective across multiple models and can be applied to multiple instructions simultaneously, demonstrating scalability and practicality for fine-grained control in language generation.

### 7. [On Debiasing Text Embeddings Through Context Injection](https://arxiv.org/pdf/2410.12874)

**Summary**: The paper investigates the biases in text embeddings and explores the potential of context injection as a debiasing method. It finds that higher-performing embedding models are more susceptible to biases but better at incorporating context, though they struggle with neutral semantics. The study introduces a dynamic top-k retrieval algorithm to mitigate biased outcomes in retrieval tasks.

### 8. [Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging](https://arxiv.org/pdf/2410.12937)

**Summary**: The paper explores a cost-effective method for adding new skills to language models by training on new skills separately and then merging them with the existing model, rather than retraining the entire model. The approach, tested on scientific literature, safety, and coding tasks, shows comparable effectiveness to full retraining while being significantly cheaper and particularly beneficial for enhancing safety features without compromising the model's ability to refuse harmful prompts.

### 9. [Interpreting token compositionality in LLMs: A robustness analysis](https://arxiv.org/pdf/2410.12924)

**Summary**: The paper introduces Constituent-Aware Pooling (CAP) to analyze how LLMs handle compositional linguistic structures, revealing that no specific layer integrates tokens into unified semantic representations. The study finds that larger models exhibit more fragmented information processing, suggesting limitations in current transformer architectures for compositional semantics and the need for new approaches in LLM design.

### 10. [Navigating the Cultural Kaleidoscope: A Hitchhiker's Guide to Sensitivity in Large Language Models](https://arxiv.org/pdf/2410.12880)

**Summary**: The paper addresses the critical issue of cultural sensitivity in LLMs, particularly in smaller models that may lack comprehensive cultural training data. It introduces a cultural harm test dataset and a culturally aligned preference dataset to evaluate and fine-tune LLMs, ensuring they respect diverse cultural norms. The study demonstrates that incorporating culturally aligned feedback significantly improves model behavior, making LLMs more inclusive and ethically sound in global applications.

### 11. [BenchmarkCards: Large Language Model and Risk Reporting](https://arxiv.org/pdf/2410.12974)

**Summary**: The paper introduces BenchmarkCards, a structured framework for documenting key properties of large language model (LLM) benchmarks, addressing the lack of standardization in the rapidly growing field. By focusing on critical characteristics such as targeted risks and evaluation methodologies, BenchmarkCards enhance transparency and reproducibility, aiding researchers in selecting appropriate benchmarks for pre-deployment evaluations.

### 12. [Self-Pluralising Culture Alignment for Large Language Models](https://arxiv.org/pdf/2410.12971)

**Summary**: The paper introduces CultureSPA, a framework designed to align LLMs with pluralistic human values across diverse cultures. By generating culture-related questions and comparing model outputs in culture-aware and culture-unaware settings, CultureSPA identifies culture-specific instances for fine-tuning, enhancing the model's alignment with various cultures. Experiments show that CultureSPA improves cultural alignment without compromising general model performance, with further enhancements possible through advanced prompt engineering.

### 13. ["Let's Argue Both Sides": Argument Generation Can Force Small Models to Utilize Previously Inaccessible Reasoning Capabilities](https://arxiv.org/pdf/2410.12997)

**Summary**: The paper introduces Argument Generation as a method to enhance logical reasoning in LLMs, particularly when traditional chain-of-thought reasoning is insufficient. By generating arguments for each possible inference result and having the model rank them, the approach effectively forces smaller models to utilize previously inaccessible reasoning capabilities, demonstrating a significant improvement in performance without the need for complex zero-shot prompting techniques.

### 14. [POROver: Improving Safety and Reducing Overrefusal in Large Language Models with Overgeneration and Preference Optimization](https://arxiv.org/pdf/2410.12999)

**Summary**: The paper introduces POROver, a method to improve the balance between safety and usefulness in large language models by overgenerating training data and applying preference optimization. The approach significantly reduces overrefusal rates from 94.4% to 45.2% for toxic prompts and further lowers it to 15.0% with preference optimization, while maintaining high safety levels.

### 15. [When Not to Answer: Evaluating Prompts on GPT Models for Effective Abstention in Unanswerable Math Word Problems](https://arxiv.org/pdf/2410.13029)

**Summary**: The paper investigates the ability of GPT models to abstain from answering unanswerable math word problems, using the UWMP dataset and evaluating performance based on abstention, correctness, and confidence. The study reveals significant limitations in GPT models' ability to handle unsolvable problems, emphasizing the need for improved models that can better manage uncertainty and complex reasoning in math word problem-solving scenarios.

### 16. [Data Defenses Against Large Language Models](https://arxiv.org/pdf/2410.13138)

**Summary**: The paper introduces "data defenses," a novel strategy enabling data owners to prevent LLMs from inferring sensitive information from their data. By automatically generating adversarial prompt injections, the method significantly reduces LLMs' ability to extract personally identifying or copyrighted information. The authors argue that this approach supports data ownership and democratic control over AI systems, and they provide evidence of its effectiveness across various LLMs and attack scenarios.

### 17. [The Geometry of Numerical Reasoning: Language Models Compare Numeric Properties in Linear Subspaces](https://arxiv.org/pdf/2410.13194)

**Summary**: The paper explores how LLMs use numerical attributes encoded in low-dimensional subspaces of the embedding space for logical comparison tasks. By identifying these subspaces through partial least squares regression and manipulating hidden states, the study demonstrates that LLMs rely on linearly encoded numerical information to make comparison decisions across various attributes.

### 18. [Mapping Bias in Vision Language Models: Signposts, Pitfalls, and the Road Ahead](https://arxiv.org/pdf/2410.13146)

**Summary**: The paper examines demographic biases in Vision Language Models (VLMs) using various datasets, finding that portrait datasets like UTKFace and CelebA are effective for bias detection, while scene-based datasets are less useful. The study introduces a more challenging version of VisoGender to improve evaluation rigor and calls for better-designed datasets to ensure VLM fairness.

### 19. [Measuring Free-Form Decision-Making Inconsistency of Language Models in Military Crisis Simulations](https://arxiv.org/pdf/2410.13204)

**Summary**: The paper investigates the inconsistency of language models (LMs) in military crisis simulations by measuring free-form decision-making responses using a BERTScore-based metric. It finds that all tested LMs exhibit semantic inconsistencies, even under different settings and prompt variations, suggesting caution in using LMs for high-stakes decision-making like military deployments.

### 20. [FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs](https://arxiv.org/pdf/2410.13210)

**Summary**: The paper introduces FaithBench, a comprehensive benchmark for evaluating hallucinations in summarization tasks by modern LLMs. It includes summaries generated by 10 LLMs from 8 different families, annotated by human experts, and highlights the challenges faced by state-of-the-art hallucination detection models, which achieve near 50% accuracy on the benchmark.

### 21. [SPIN: Self-Supervised Prompt INjection](https://arxiv.org/pdf/2410.13236)

**Summary**: The paper introduces SPIN, a self-supervised prompt injection technique designed to enhance the safety and reliability of Large Language Models (LLMs) by detecting and mitigating adversarial attacks. SPIN reduces the attack success rate by up to 87.9% while preserving performance on benign queries, and it remains effective even against adaptive attackers aware of the defense mechanism.

### 22. [Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis](https://arxiv.org/pdf/2410.13237)

**Summary**: The paper introduces Language Confusion Entropy, a new metric to quantify the phenomenon where Large Language Models (LLMs) generate text in unintended or inappropriate languages. By analyzing linguistic typology and lexical variation, the study identifies patterns of language confusion across LLMs and links this issue to security vulnerabilities, particularly in multilingual embedding inversion attacks. The findings suggest that understanding linguistic typology can enhance LLM alignment and security.

### 23. [Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning](https://arxiv.org/pdf/2410.13274)

**Summary**: The paper identifies a critical limitation in current unlearning techniques for LLMs, where multi-hop queries can still retain indirect knowledge even after unlearning intermediate steps. To address this, the authors propose MUNCH, an uncertainty-based approach that decomposes multi-hop queries into subquestions and uses model uncertainty to enhance unlearning effectiveness. Empirical results show that MUNCH significantly improves the removal of indirect knowledge and can be seamlessly integrated with existing unlearning methods.

### 24. [BANTH: A Multi-label Hate Speech Detection Dataset for Transliterated Bangla](https://arxiv.org/pdf/2410.13281)

**Summary**: The paper introduces BanTH, the first multi-label hate speech detection dataset for transliterated Bangla, containing 37.3k samples from YouTube comments labeled with multiple target groups. The authors establish state-of-the-art baselines using transformer encoders pre-trained on transliterated Bangla and propose a novel translation-based prompting strategy for zero-shot performance, addressing a critical gap in hate speech research for underrepresented languages.

### 25. [Mitigating Biases to Embrace Diversity: A Comprehensive Annotation Benchmark for Toxic Language](https://arxiv.org/pdf/2410.13313)

**Summary**: The paper presents a prescriptive annotation benchmark for labeling toxic language, emphasizing consistency and reducing bias by incorporating humanities research. It introduces two new datasets with improved inter-annotator agreement between human and language model annotations, demonstrating that smaller models fine-tuned on multi-source data can outperform larger models trained on single-source human annotations. This approach underscores the importance of structured guidelines in maintaining performance and embracing language diversity.

### 26. [Do LLMs Overcome Shortcut Learning? An Evaluation of Shortcut Challenges in Large Language Models](https://arxiv.org/pdf/2410.13343)

**Summary**: The paper introduces Shortcut Suite, a test suite to assess how Large Language Models (LLMs) handle dataset biases as shortcuts, revealing that LLMs often rely on these shortcuts, impairing performance. Key findings include larger LLMs being more prone to shortcut use, chain-of-thought prompting reducing shortcut reliance, and LLMs showing overconfidence and lower explanation quality on shortcut-laden datasets.

### 27. [Do LLMs Have Political Correctness? Analyzing Ethical Biases and Jailbreak Vulnerabilities in AI Systems](https://arxiv.org/pdf/2410.13334)

**Summary**: The paper investigates the ethical biases introduced into LLMs to ensure their safety, revealing that these biases can be exploited through "PCJailbreak" techniques, which show significant differences in jailbreak success rates based on gender and racial keywords. The authors propose PCDefense as an efficient defense method against such exploits, emphasizing the need for more responsible safety measures in LLM development.

### 28. [Enhancing Text Generation in Joint NLG/NLU Learning Through Curriculum Learning, Semi-Supervised Training, and Advanced Optimization Techniques](https://arxiv.org/pdf/2410.13498)

**Summary**: This paper introduces an advanced approach to enhance text generation in joint NLG/NLU learning by integrating curriculum learning, semi-supervised training, and sophisticated optimization techniques. The model leverages transformer-based architectures, pre-trained language models, and reinforcement learning methods to improve coherence, diversity, and contextual relevance in generated text.

### 29. [Bias in the Mirror : Are LLMs opinions robust to their own adversarial attacks ?](https://arxiv.org/pdf/2410.13517)

**Summary**: The paper investigates the robustness of biases in LLMs by having two instances of an LLM debate opposing viewpoints to persuade a neutral version. The study examines how biases persist and whether models reinforce misinformation or shift to harmful viewpoints across different LLMs, sizes, origins, and languages.

### 30. [Seeing Through VisualBERT: A Causal Adventure on Memetic Landscapes](https://arxiv.org/pdf/2410.13488)

**Summary**: The paper introduces a Structural Causal Model (SCM) framework to enhance the interpretability of VisualBERT in detecting offensive memes, addressing challenges with non-causal attributions and implicit offensive content. The framework allows for transparent interpretation of model behavior and identifies reasons behind misclassifications, with quantitative analysis showing that input attribution methods do not ensure causality, raising concerns for safety-critical applications.

### 31. [A new approach for fine-tuning sentence transformers for intent classification and out-of-scope detection tasks](https://arxiv.org/pdf/2410.13649)

**Summary**: The paper introduces a novel approach to fine-tuning sentence transformers for intent classification and out-of-scope detection by incorporating an in-scope embedding reconstruction loss through an auto-encoder, which helps to reduce the overlap between in-scope and out-of-scope embeddings. This method improves out-of-sample rejection performance by 1-4% in the area under the precision-recall curve without affecting intent classification accuracy.

### 32. [SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs](https://arxiv.org/pdf/2410.13648)

**Summary**: The paper introduces SimpleToM, a dataset designed to test whether LLMs can implicitly apply "theory of mind" (ToM) reasoning to predict behavior and judge rationality in social scenarios. Despite LLMs' ability to predict mental states, they often fail at predicting subsequent behavior and judging its rationality, even when aware of the protagonist's mental state. The study suggests that while models can be improved with specific interventions, their natural performance remains limited, highlighting potential challenges in deploying LLMs for social interactions.

### 33. [PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment](https://arxiv.org/pdf/2410.13785)

**Summary**: The paper introduces PopAlign, a framework that enhances the alignment of large language models by diversifying contrasting patterns across prompts, models, and pipelines. By integrating six contrasting strategies, PopAlign improves the comprehensiveness of alignment and demonstrates superior performance compared to traditional methods, making models less susceptible to jailbreaking attacks.

### 34. [On the Role of Attention Heads in Large Language Model Safety](https://arxiv.org/pdf/2410.13708)

**Summary**: The paper investigates the role of attention heads in the safety capabilities of LLMs, introducing a novel metric called Safety Head ImPortant Score (Ships) to assess individual heads' contributions to safety. The study finds that specific attention heads significantly impact safety, with ablating a single safety head leading to a substantial increase in harmful responses, suggesting that attention heads primarily function as safety feature extractors.

### 35. [Looking Inward: Language Models Can Learn About Themselves by Introspection](https://arxiv.org/pdf/2410.13787)

**Summary**: The paper explores whether LLMs can introspect, acquiring knowledge about their internal states that is not derived from training data. Through experiments with GPT-4, GPT-4o, and Llama-3 models, the authors find that a model can predict its own behavior better than another model trained on its behavior, suggesting introspective capabilities. However, this ability is limited to simpler tasks and does not generalize well to more complex or out-of-distribution scenarios.

### 36. [A Watermark for Order-Agnostic Language Models](https://arxiv.org/pdf/2410.13805)

**Summary**: The paper introduces Pattern-mark, a novel watermarking framework designed for order-agnostic language models (LMs), which addresses the challenge of non-sequential token generation. By using a Markov-chain-based watermark generator and a statistical pattern-based detection algorithm, Pattern-mark enhances detection efficiency, generation quality, and robustness, outperforming existing techniques in extensive evaluations on models like ProteinMPNN and CMLM.

### 37. [De-mark: Watermark Removal in Large Language Models](https://arxiv.org/pdf/2410.13808)

**Summary**: The paper introduces De-mark, a framework for removing n-gram-based watermarks from large language models. By employing a novel querying strategy called random selection probing, De-mark assesses watermark strength and identifies the red-green list, effectively removing the watermark. Experiments on models like Llama3 and ChatGPT show De-mark's efficiency and effectiveness in watermark removal tasks.

### 38. [Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization](https://arxiv.org/pdf/2410.12949)

**Summary**: The paper explores how mechanistic interpretability can enhance the precision and effectiveness of knowledge editing and unlearning in large language models. It distinguishes between methods that focus on preserving outputs and those that identify high-level mechanisms with predictable intermediate states, finding that the latter leads to more robust unlearning and editing, particularly when targeting the lookup-table mechanism for factual recall. This approach reduces unintended side effects and resists attempts to relearn unwanted information, outperforming baseline methods across various datasets and models.

### 39. [Sensitivity of Generative VLMs to Semantically and Lexically Altered Prompts](https://arxiv.org/pdf/2410.13030)

**Summary**: The paper investigates the sensitivity of generative vision-language models (VLMs) to lexical and semantic alterations in prompts using the SugarCrepe++ dataset. It finds that VLMs are highly sensitive to lexical changes without corresponding semantic shifts, which impacts the effectiveness of techniques designed to ensure consistent output.

### 40. [Controllable Generation via Locally Constrained Resampling](https://arxiv.org/pdf/2410.13111)

**Summary**: The paper introduces a probabilistic approach for generating constrained outputs from autoregressive models by locally resampling sequences to satisfy specific constraints. The method outperforms existing techniques in tasks like LLM detoxification and Sudoku puzzle solving, demonstrating its effectiveness in generating outputs that adhere to complex logical requirements.

### 41. [Self-Comparison for Dataset-Level Membership Inference in Large (Vision-)Language Models](https://arxiv.org/pdf/2410.13088)

**Summary**: The paper introduces a novel dataset-level membership inference method called Self-Comparison, which leverages paraphrasing to detect memorization in LLMs and vision-language models (VLMs). By comparing the likelihood of sequences before and after paraphrasing, the method can infer membership without needing ground-truth member or non-member data, outperforming traditional membership inference attacks across various datasets and models.

### 42. [Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding](https://arxiv.org/pdf/2410.13321)

**Summary**: The paper introduces Summary-Guided Decoding (SGD) to mitigate hallucinations in Large Vision-Language Models (LVLMs) by reducing reliance on language priors. SGD focuses on image-related part-of-speech tokens while maintaining text quality, achieving state-of-the-art performance on object hallucination benchmarks and demonstrating a robust balance between precision and recall.

### 43. [A Common Pitfall of Margin-based Language Model Alignment: Gradient Entanglement](https://arxiv.org/pdf/2410.13828)

**Summary**: The paper identifies a critical issue in margin-based language model alignment, termed **gradient entanglement**, where the optimization of preferred and dispreferred responses becomes coupled, leading to unintended increases in unsafe responses and decreases in preferred ones. The authors provide theoretical and empirical evidence to explain this phenomenon and suggest potential algorithm modifications to mitigate its effects, thereby improving the alignment of language models.

### 44. [TAIA: Large Language Models are Out-of-Distribution Data Learners](https://arxiv.org/pdf/2405.20192)

**Summary**: The paper introduces TAIA (\trainallInfAttn), an inference-time intervention method that selectively uses only fine-tuned attention parameters from the Transformer architecture, enhancing performance in data-scarce domains with domain-mismatched data. Empirical results across various tasks and model sizes show that TAIA significantly outperforms fully fine-tuned models and base models, particularly in scenarios with data distribution mismatches.

### 45. [Avoiding Copyright Infringement via Large Language Model Unlearning](https://arxiv.org/pdf/2406.10952)

**Summary**: The paper introduces Stable Sequential Unlearning (SSU), a novel framework for removing copyrighted content from Large Language Models (LLMs) over multiple time steps. SSU identifies and removes specific weight updates related to copyrighted material while maintaining the model's general language capabilities, demonstrating superior performance compared to existing methods in balancing unlearning efficacy and language proficiency.

### 46. [Modeling Human Subjectivity in LLMs Using Explicit and Implicit Human Factors in Personas](https://arxiv.org/pdf/2406.14462)

**Summary**: The paper explores the use of explicit and implicit human factors in personas to model human subjectivity in LLMs for social scientific tasks. It finds that while explicit personas show mixed results in reproducing human biases, LLMs generally fail to capture implicit biases, suggesting limitations in modeling complex human perceptions and interactions.

### 47. [Evaluating Evidence Attribution in Generated Fact Checking Explanations](https://arxiv.org/pdf/2406.12645)

**Summary**: The paper introduces a novel evaluation protocol, citation masking and recovery, to assess the quality of evidence attribution in automated fact-checking explanations. It finds that while LLMs can partially automate attribution assessment, they still produce explanations with inaccurate attributions, highlighting the necessity of human-curated evidence for improved fact-checking.

### 48. [Unmasking Database Vulnerabilities: Zero-Knowledge Schema Inference Attacks in Text-to-SQL Systems](https://arxiv.org/pdf/2406.14545)

**Summary**: The paper introduces a zero-knowledge framework for reconstructing database schemas in text-to-SQL systems, demonstrating significant security vulnerabilities. By probing the system with crafted questions and using GPT-4 as an interpreter, the method achieves high accuracy in identifying schema elements, raising concerns about unauthorized data access. The study also proposes a protection mechanism, though it shows limited effectiveness in mitigating these attacks.

### 49. [Evaluating Fairness in Large Vision-Language Models Across Diverse Demographic Attributes and Prompts](https://arxiv.org/pdf/2406.17974)

**Summary**: The paper evaluates the fairness of large vision-language models (LVLMs) across various demographic attributes using public benchmark datasets. It finds that despite advancements, these models still exhibit fairness issues, particularly in zero-shot prompting scenarios. The study proposes a multi-modal Chain-of-thought (CoT) strategy as a potential method to mitigate biases, enhancing transparency and scalability in addressing fairness concerns.

### 50. [InferAct: Inferring Safe Actions for LLM-Based Agents Through Preemptive Evaluation and Human Feedback](https://arxiv.org/pdf/2407.11843)

**Summary**: The paper introduces InferAct, a method that uses the belief reasoning capabilities of LLMs to preemptively evaluate and detect potential risky actions in LLM-based agents, ensuring safer operations. By acting as a human proxy, InferAct alerts users to unsafe actions, preventing irreversible risks and improving decision-making in critical environments.

### 51. [Understanding and Mitigating Language Confusion in LLMs](https://arxiv.org/pdf/2406.20052)

**Summary**: The paper investigates the limitation of LLMs in consistently generating text in the user's desired language, introducing the Language Confusion Benchmark (LCB) to evaluate this issue across 15 languages. The study finds that models like Llama Instruct and Mistral exhibit significant language confusion, especially with complex prompts and high sampling temperatures, but suggests that this can be mitigated through few-shot prompting, multilingual supervised fine-tuning, and preference tuning. The LCB is released as a tool for scalable multilingual evaluation.

### 52. [Granular Privacy Control for Geolocation with Vision Language Models](https://arxiv.org/pdf/2407.04952)

**Summary**: The paper highlights the privacy risks associated with Vision Language Models (VLMs) in geolocating images, demonstrating that current VLMs are highly effective in this task. To address these risks, the authors introduce GPTGeoChat, a benchmark for evaluating VLMs' ability to moderate geolocation dialogues, and find that fine-tuned models are effective at identifying location information leaks at broader levels, but require supervised data for more precise moderation.

### 53. [Larger Language Models Don't Care How You Think: Why Chain-of-Thought Prompting Fails in Subjective Tasks](https://arxiv.org/pdf/2409.06173)

**Summary**: The paper investigates the effectiveness of Chain-of-Thought (CoT) prompting in Large Language Models (LLMs) for subjective tasks, finding that CoT often relies on retrieving reasoning priors rather than adapting to the task. This leads to a phenomenon similar to posterior collapse, where the model's reasoning process remains relatively unchanged despite the input, limiting its performance in complex subjective domains.

### 54. [REAL: Response Embedding-based Alignment for LLMs](https://arxiv.org/pdf/2409.17169)

**Summary**: The paper introduces REAL, a method for efficiently aligning LLMs to human preferences by focusing on selecting dissimilar response pairs for labeling. This approach reduces the labeling workload by up to 65% while improving the alignment quality, as demonstrated by experiments on synthetic and real-world datasets.

### 55. [PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action](https://arxiv.org/pdf/2409.00138)

**Summary**: The paper introduces PrivacyLens, a framework for evaluating the privacy norm awareness of language models (LMs) by extending privacy-sensitive scenarios into detailed vignettes and agent trajectories. The study finds that even state-of-the-art LMs like GPT-4 and Llama-3-70B exhibit significant privacy leaks, highlighting the need for more robust privacy-aware models. The framework's dynamic nature allows for the exploration of various scenarios to identify and mitigate privacy risks in LM-mediated communication.

### 56. [Temporally Consistent Factuality Probing for Large Language Models](https://arxiv.org/pdf/2409.14065)

**Summary**: The paper introduces TeCFaP, a new task to evaluate the temporal consistency of factuality in Large Language Models (LLMs), addressing limitations in existing benchmarks. It proposes TEMP-COFAC, a dataset for this task, and extends existing metrics to measure temporal consistency. The study finds that LLMs generally perform poorly on TeCFaP and introduces CoTSeLF, a framework combining multi-task instruction tuning and consistent-time-sensitive reinforcement learning, to improve performance.

### 57. [SafeGen: Mitigating Sexually Explicit Content Generation in Text-to-Image Models](https://arxiv.org/pdf/2404.06666)

**Summary**: The paper introduces SafeGen, a framework designed to prevent text-to-image models from generating sexually explicit content by eliminating explicit visual representations regardless of the text input. SafeGen effectively mitigates such content generation, outperforming existing methods and achieving a 99.4% removal rate, while maintaining the quality of benign images.

### 58. [On the Reliability of Large Language Models to Misinformed and Demographically-Informed Prompts](https://arxiv.org/pdf/2410.10850)

**Summary**: The paper examines the performance of Large Language Model-backed chatbots in responding to misinformed prompts and questions with demographic information in the domains of Climate Change and Mental Health. It finds that while chatbots perform well on True/False questions, there are significant concerns regarding privacy, ethics, and the need for directing users to professional services. The study concludes that careful consideration, ethical oversight, and rigorous refinement are necessary for the deployment of these chatbots in sensitive areas.

### 59. [Negative-Prompt-driven Alignment for Generative Language Model](https://arxiv.org/pdf/2410.12194)

**Summary**: The paper introduces NEAT, a method that uses negative prompts to align generative language models with human values by penalizing undesirable outputs. By incorporating both positive and negative examples into the training process, NEAT improves the model's ability to avoid harmful or biased responses, enhancing overall alignment with human preferences.

### 60. [Iter-AHMCL: Alleviate Hallucination for Large Language Model via Iterative Model-level Contrastive Learning](https://arxiv.org/pdf/2410.12130)

**Summary**: The paper introduces Iterative Model-level Contrastive Learning (Iter-AHMCL) to reduce hallucinations in Large Language Models (LLMs) by modifying representation layers using contrastive models trained on data with and without hallucinations. This approach achieves an average improvement of 10.1 points on the TruthfulQA benchmark across four LLMs, demonstrating its effectiveness in reducing hallucination while preserving model capabilities.

### 61. [Exploring Large Language Models for Hate Speech Detection in Rioplatense Spanish](https://arxiv.org/pdf/2410.12174)

**Summary**: The paper investigates the performance of LLMs like ChatGPT 3.5, Mixtral, and Aya in detecting hate speech in Rioplatense Spanish, comparing them to a fine-tuned BERT classifier. While LLMs exhibit lower precision and may struggle with certain slang or slurs, they demonstrate sensitivity to nuanced hate speech, particularly homophobic and transphobic content. The study emphasizes the importance of tailored corpora for effective hate speech detection and makes its resources available for further research.

### 62. [Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors](https://arxiv.org/pdf/2410.12299)

**Summary**: The paper introduces Semantics-Adaptive Dynamic Intervention (SADI), a novel method for modifying the behavior of LLMs by dynamically adjusting activation vectors based on input semantics. SADI outperforms existing methods by identifying and scaling critical model elements during inference, enhancing task performance without additional training, and demonstrating versatility across different LLM architectures and tasks.

### 63. [Open Domain Question Answering with Conflicting Contexts](https://arxiv.org/pdf/2410.12311)

**Summary**: The paper introduces the Question Answering with Conflicting Contexts (QACC) dataset, highlighting that up to 25% of open domain questions yield conflicting information when retrieved from sources like Google Search. The study evaluates three Large Language Models (LLMs) and shows their limitations in handling such conflicts. By fine-tuning LLMs to explain their answers, the authors suggest a potential improvement in their ability to reason through conflicting contexts.

### 64. [Neuron-based Personality Trait Induction in Large Language Models](https://arxiv.org/pdf/2410.12327)

**Summary**: The paper introduces a neuron-based approach for inducing personality traits in LLMs, focusing on the Big Five personality traits. It contributes by creating PersonalityBench, a dataset for evaluating LLM personality traits, and proposes methods to identify and manipulate personality-related neurons within LLMs. This approach allows for fine-grained control over LLM traits without retraining, achieving performance comparable to fine-tuned models.

### 65. [A linguistic analysis of undesirable outcomes in the era of generative AI](https://arxiv.org/pdf/2410.12341)

**Summary**: The paper examines the linguistic degradation in generative AI models, particularly LLama2, showing that repeated fine-tuning on self-generated content leads to reduced lexical richness and distorted linguistic patterns. This "model collapse" not only diminishes content diversity but also introduces inaccuracies and biases, emphasizing the need for careful input curation to mitigate these issues.

### 66. [ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs](https://arxiv.org/pdf/2410.12405)

**Summary**: The paper introduces ProSA, a framework for assessing and understanding the prompt sensitivity of LLMs. It introduces a new metric, PromptSensiScore, and uses decoding confidence to analyze how prompt variations affect model performance and subjective evaluations. The study finds that prompt sensitivity varies across datasets and models, with larger models showing greater robustness, and that few-shot examples can mitigate sensitivity issues, especially in complex tasks.

### 67. [Conformity in Large Language Models](https://arxiv.org/pdf/2410.12428)

**Summary**: The paper investigates the conformity bias in LLMs, finding that all tested models exhibit varying levels of conformity to majority opinions, especially when uncertain about their own predictions. The study also identifies factors influencing conformity, such as training paradigms and input characteristics, and proposes interventions like Devil's Advocate and Question Distillation to reduce this bias in LLMs.

### 68. [Retrieval-Reasoning Large Language Model-based Synthetic Clinical Trial Generation](https://arxiv.org/pdf/2410.12476)

**Summary**: The paper introduces a Retrieval-Reasoning framework using LLMs to generate synthetic clinical trials, addressing challenges like data scarcity and ethical concerns. The synthetic trials, with binary success/failure labels, effectively augment real datasets, improving model training for trial outcome prediction. This approach shows potential for accelerating clinical research while maintaining patient privacy.

### 69. [With a Grain of SALT: Are LLMs Fair Across Social Dimensions?](https://arxiv.org/pdf/2410.12499)

**Summary**: The paper investigates biases in open-source Large Language Models (LLMs) across gender, religion, and race by introducing a bias detection dataset generated using seven bias triggers. The study evaluates Llama and Gemma models, finding consistent polarization toward certain groups, with language variations revealing cultural and contextual influences on bias manifestation.

### 70. [Advancing Fairness in Natural Language Processing: From Traditional Methods to Explainability](https://arxiv.org/pdf/2410.12511)

**Summary**: This PhD thesis explores the integration of fairness and explainability in Natural Language Processing (NLP), introducing innovative algorithms to mitigate biases in multi-class classifiers and Transformer models. It also critiques standard fairness metrics and proposes new methods like COCKATIEL and TaCo to enhance transparency and equity in NLP systems, contributing to the broader discourse on responsible AI development.

### 71. [Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse RL](https://arxiv.org/pdf/2410.12491)

**Summary**: The paper presents a novel approach to interpreting LLMs trained with Reinforcement Learning from Human Feedback (RLHF) by applying inverse reinforcement learning (IRL) to recover their implicit reward functions. The study finds that IRL-derived reward models can predict human preferences with high accuracy and be used to fine-tune new LLMs, offering insights into the non-identifiability of reward functions and the relationship between model size and interpretability. This work enhances understanding of LLM alignment and has implications for the responsible development of these models.

### 72. [On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs](https://arxiv.org/pdf/2410.12600)

**Summary**: The paper investigates the risk of evidence pollution in malicious social text detection due to the rise of LLMs, which can manipulate evidence to confuse detectors. It proposes three defense strategies to mitigate this risk but highlights practical limitations, such as the need for annotated data and high inference costs. The study concludes that polluted evidence, especially when generated by LLMs, significantly compromises detection models and can amplify negative impacts.

### 73. [Can We Reverse In-Context Knowledge Edits?](https://arxiv.org/pdf/2410.12586)

**Summary**: The paper investigates the detection and reversal of in-context knowledge edits (IKE) in LLMs, which can be misused to insert harmful content. The authors demonstrate high accuracy in detecting IKE-edits using top-10 token probabilities and introduce a method to reverse these edits using specially tuned reversal tokens, enhancing LLM resilience and transparency.

### 74. [Weak-to-Strong Generalization beyond Accuracy: a Pilot Study in Safety, Toxicity, and Legal Reasoning](https://arxiv.org/pdf/2410.12621)

**Summary**: The paper explores the application of weak-to-strong generalization in practical alignment tasks such as safety, toxicity, and legal reasoning, addressing the limitations of traditional human feedback-based methods for superhuman language models. It demonstrates the effectiveness of this approach in complex alignment scenarios and proposes strategies to improve alignment performance, aiming to advance research in this area.

### 75. [Building Better: Avoiding Pitfalls in Developing Language Resources when Data is Scarce](https://arxiv.org/pdf/2410.12691)

**Summary**: The paper examines the challenges of developing language resources for mid- to low-resource languages, focusing on issues related to data quality and ethical annotation practices. Through feedback from stakeholders, it identifies concerns such as linguistic and cultural appropriateness of data and the misuse of online communities for annotation. The study concludes with recommendations for creating high-quality, culturally sensitive language resources that respect the dignity of data workers.

### 76. [Unitary Multi-Margin BERT for Robust Natural Language Processing](https://arxiv.org/pdf/2410.12759)

**Summary**: The paper introduces UniBERT, a novel approach to enhance the robustness of BERT models against adversarial attacks by combining unitary weights with multi-margin loss. This method significantly improves post-attack classification accuracy by up to 73.8% while maintaining competitive pre-attack performance, with the tradeoff between pre- and post-attack accuracy adjustable via a single parameter.

### 77. [WorldMedQA-V: a multilingual, multimodal medical examination dataset for multimodal language models evaluation](https://arxiv.org/pdf/2410.12722)

**Summary**: The paper introduces WorldMedQA-V, a multilingual, multimodal medical examination dataset designed to evaluate vision language models (VLMs) in healthcare. The dataset includes 568 multiple-choice questions paired with medical images from four countries, with questions available in both original languages and validated English translations. The authors provide baseline performance metrics for various models, highlighting the importance of this benchmark for ensuring the safety and efficacy of VLMs in diverse healthcare settings.

### 78. [Boosting Logical Fallacy Reasoning in LLMs via Logical Structure Tree](https://arxiv.org/pdf/2410.12048)

**Summary**: The paper introduces a novel approach to enhance logical fallacy reasoning in LLMs by constructing a logical structure tree that explicitly represents the hierarchical logic flow within statements. This tree, built using unsupervised methods and guided by constituency trees and a taxonomy of connectives, is integrated into LLMs through both textual and embedding-based prompts. The approach significantly improves the precision and recall of fallacy detection and classification in benchmark datasets.

### 79. [Concept-Reversed Winograd Schema Challenge: Evaluating and Improving Robust Reasoning in Large Language Models via Abstraction](https://arxiv.org/pdf/2410.12040)

**Summary**: The paper introduces the Concept-Reversed Winograd Schema Challenge (CR-WSC) to test the robustness of Large Language Models (LLMs) in reasoning, finding that reversing concepts associated with wrong answers significantly reduces their performance. Additionally, the authors propose Abstraction-of-Thought (AoT), a method that uses conceptual abstraction to enhance LLMs' consistency and robustness in reasoning, as evidenced by experiments on the CR-WSC dataset.

### 80. [Bias Similarity Across Large Language Models](https://arxiv.org/pdf/2410.12010)

**Summary**: The paper investigates bias similarity across ten Large Language Models (LLMs) from four model families, finding that fine-tuning does not effectively mitigate bias, models within the same family exhibit dissimilar output distributions, and there is a risk of training data information leakage. The study highlights the challenges in addressing bias in LLMs and raises concerns about their real-world deployment.

### 81. [Preference Optimization with Multi-Sample Comparisons](https://arxiv.org/pdf/2410.12138)

**Summary**: The paper introduces a new approach to post-training optimization for generative models, specifically addressing the limitations of single-sample comparisons in methods like RLHF and DAP. By proposing Multi-sample Direct Preference Optimization (mDPO) and Multi-sample Identity Preference Optimization (mIPO), the authors demonstrate that multi-sample comparisons are more effective in capturing generative diversity and bias, leading to a more robust optimization framework, especially in the presence of label noise.

### 82. [Controlled Automatic Task-Specific Synthetic Data Generation for Hallucination Detection](https://arxiv.org/pdf/2410.12278)

**Summary**: The paper introduces a novel method for generating task-specific synthetic datasets for hallucination detection, featuring a two-step pipeline with hallucination pattern guidance and language style alignment. The approach improves the generalization and robustness of hallucination detectors, outperforming in-context-learning-based detectors by 32%. The data mixture strategy further enhances the performance across different tasks and generators.

### 83. [OmnixR: Evaluating Omni-modality Language Models on Reasoning across Modalities](https://arxiv.org/pdf/2410.12219)

**Summary**: The paper introduces OmnixR, an evaluation suite for benchmarking Omni-modality Language Models (OLMs) like GPT-4o and Gemini, which are designed to handle multiple modalities (text, vision, audio). OmnixR addresses the lack of comprehensive multi-modal assessments by offering synthetic and realistic subsets, challenging models to reason across diverse modalities. The study reveals that current state-of-the-art OLMs struggle with tasks requiring integrated multi-modal reasoning, highlighting the need for improved alignment in omni-modal AI systems.

### 84. [Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models](https://arxiv.org/pdf/2410.12662)

**Summary**: The paper identifies a vulnerability in Large Vision-Language Models (LVLMs) where the safety mechanisms designed for text are not effectively transferred to visual inputs. The authors propose a Text-Guided vision-language Alignment method (TGA) that uses related text to guide the projection of visual data into the hidden states space, ensuring the safety mechanism is correctly applied to images. This approach maintains both safety and performance across various vision tasks.

### 85. [CREAM: Consistency Regularized Self-Rewarding Language Models](https://arxiv.org/pdf/2410.12735)

**Summary**: The paper introduces CREAM, a Consistency Regularized Self-Rewarding Language Model, to address the issue of accumulated bias in self-rewarding language models, which can lead to unreliable preference data. By leveraging rewarding consistency across iterations, CREAM helps the model learn from more reliable data, resulting in improved reward consistency and alignment performance.

### 86. [TaCo: Targeted Concept Erasure Prevents Non-Linear Classifiers From Detecting Protected Attributes](https://arxiv.org/pdf/2312.06499)

**Summary**: The paper introduces Targeted Concept Erasure (TaCo), a method designed to remove sensitive information from latent representations in NLP models, ensuring fairness against non-linear classifiers. TaCo outperforms existing methods by significantly reducing the prediction accuracy of sensitive attributes while maintaining overall task performance.

### 87. [UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models](https://arxiv.org/pdf/2402.10052)

**Summary**: The paper introduces UnDIAL, a robust unlearning method for large language models that uses self-distillation with adjusted logits to selectively reduce the influence of targeted tokens, thereby enhancing privacy and safety. Unlike existing methods, UnDIAL ensures stable convergence and avoids over-unlearning, demonstrating robustness and scalability across various unlearning tasks.

### 88. [ToBlend: Token-Level Blending With an Ensemble of LLMs to Attack AI-Generated Text Detection](https://arxiv.org/pdf/2402.11167)

**Summary**: The paper introduces ToBlend, a token-level ensemble method that combines outputs from multiple LLMs to generate text that can evade AI-content detection systems. By randomly selecting tokens from different LLMs, ToBlend significantly reduces the accuracy of current detection methods. The study also fine-tunes a Llama3.1 model to better identify ToBlend-generated text, highlighting the potential of such adversarial techniques to improve detection robustness.

### 89. [Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](https://arxiv.org/pdf/2410.12777)

**Summary**: The paper introduces meta-unlearning for diffusion models (DMs) to prevent the relearning of unlearned harmful or copyrighted concepts through malicious finetuning. The proposed method ensures that benign concepts related to unlearned ones self-destruct when exposed to such finetuning, thereby maintaining the integrity of the unlearning process. The approach is validated on Stable Diffusion models and is shown to be effective and compatible with existing unlearning techniques.

### 90. [A Watermark for Low-entropy and Unbiased Generation in Large Language Models](https://arxiv.org/pdf/2405.14604)

**Summary**: The paper introduces the Sampling One Then Accepting (STA-1) method, a novel watermarking technique for LLMs that addresses several limitations of previous unbiased watermarking methods, including the need for white-box access, long detection times, and lack of robustness against attacks. The study highlights the tradeoff between watermark strength and text quality, particularly in low-entropy scenarios, and demonstrates that STA-1 achieves comparable text quality and watermark strength with a reduced risk of unsatisfactory outputs.

### 91. [The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences of LLM Evaluators](https://arxiv.org/pdf/2406.12319)

**Summary**: The paper highlights that LLM evaluators exhibit biased preferences, particularly when using pairwise comparisons, which amplify superficial attributes like verbosity. To mitigate this, the authors propose PRePair, a method that combines pointwise reasoning within a pairwise framework, enhancing unbiased evaluation and improving performance on both adversarial and standard benchmarks.

### 92. [MFC-Bench: Benchmarking Multimodal Fact-Checking with Large Vision-Language Models](https://arxiv.org/pdf/2406.11288)

**Summary**: The paper introduces MFC-Bench, a benchmark for evaluating the factual accuracy of large vision-language models (LVLMs) in multimodal fact-checking. It assesses models across three stages of verdict prediction and reveals that current LVLMs struggle with detecting manipulated content, highlighting the need for more reliable AI systems. The benchmark and resources are made publicly available to support further research in this area.

### 93. [CELL your Model: Contrastive Explanations for Large Language Models](https://arxiv.org/pdf/2406.11785)

**Summary**: The paper introduces contrastive explanation methods for LLMs by identifying why an LLM produces a specific output in response to a prompt. The authors propose two algorithms: a myopic algorithm that requires many model calls and a budgeted algorithm that efficiently adheres to a query budget, both of which are shown to be effective across various natural language tasks.

### 94. [Exploring Changes in Nation Perception with Nationality-Assigned Personas in LLMs](https://arxiv.org/pdf/2406.13993)

**Summary**: The study investigates how Large Language Models (LLMs) alter their perceptions of nations when assigned specific nationality personas, finding a bias favoring Western European nations. The research highlights discrepancies between LLM evaluations and human survey responses, emphasizing the need for mechanisms to ensure fairness and prevent over-generalization in LLM outputs.

### 95. [Enhancing Data Privacy in Large Language Models through Private Association Editing](https://arxiv.org/pdf/2406.18221)

**Summary**: The paper introduces Private Association Editing (PAE), a novel method to enhance data privacy in LLMs by removing Personally Identifiable Information (PII) without retraining the model. Experimental results show that PAE outperforms alternative methods, suggesting it as a crucial tool for protecting data privacy in LLMs, thereby fostering safer model development for practical applications.

### 96. [Core: Robust Factual Precision with Informative Sub-Claim Identification](https://arxiv.org/pdf/2407.03572)

**Summary**: The paper introduces Core, a customizable subclaim selection component designed to enhance the robustness of factual precision metrics in large language models by filtering out repetitive and non-informative subclaims. The authors demonstrate that integrating Core with existing metrics significantly improves their performance across various knowledge domains and advocate for its adoption in the community, releasing an expanded dataset and an evaluation framework to support further research.

### 97. [Measuring and Benchmarking Large Language Models' Capabilities to Generate Persuasive Language](https://arxiv.org/pdf/2406.17753)

**Summary**: The paper investigates the capability of Large Language Models (LLMs) to generate persuasive language across various domains, comparing their performance when explicitly instructed to enhance or reduce persuasion and when merely paraphrasing. The study introduces a new dataset, Persuasive-Pairs, which pairs original texts with LLM-generated rewrites, annotated on a relative scale for persuasion. The analysis reveals that different personas in LLaMA3's system prompt significantly influence the level of persuasive language, even in paraphrasing tasks.

### 98. [Beyond Instruction Following: Evaluating Inferential Rule Following of Large Language Models](https://arxiv.org/pdf/2407.08440)

**Summary**: The paper introduces RuleBench, a benchmark to evaluate the inferential rule-following capabilities of Large Language Models (LLMs), distinguishing it from mere instruction-following. The study finds that LLMs struggle with rule-following and proposes Inferential Rule-Following Tuning (IRFT) to improve this ability, showing that LLMs can learn abstract rule-following from synthetic data and apply it to RuleBench scenarios.

### 99. [LoraMap: Harnessing the Power of LoRA Connections](https://arxiv.org/pdf/2408.16264)

**Summary**: The paper introduces LoraMap, a novel approach for connecting multiple Low-Rank Adaptation (LoRA) modules in Large Language Models (LLMs) to enhance fact-checking performance. By creating specialized reasoning datasets and fine-tuning individual LoRAs, LoraMap establishes connections between these modules, outperforming existing methods like LoraHub and LoraConcat in fact-checking tasks while using fewer trainable parameters.

### 100. [MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs](https://arxiv.org/pdf/2409.02257)

**Summary**: The paper introduces MMLU-Pro+, an advanced benchmark designed to evaluate higher-order reasoning and shortcut learning in LLMs. By including questions with multiple correct answers across various domains, MMLU-Pro+ challenges LLMs to engage in complex reasoning and resist simplistic problem-solving strategies. The study reveals significant performance variations among state-of-the-art LLMs, emphasizing the need for more rigorous evaluation frameworks.



---

*Last updated on 2024-10-18*