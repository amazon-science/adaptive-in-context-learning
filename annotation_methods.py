import random
import os
from utils import reliability_plot, embedding_plot
import json
from algorithms import cluster, fast_votek_mod, uncertainty_ranking, votek_mod
from algorithms import density_max_coverage


def selective_annotation_single_phase(args,**kwargs):
    """
    Single-step annotation methods: random, fast-votek, votek, hardest, adaicl-base

    Args:
        args

    Returns:
        list: selected data points for annotation
    """

    random.seed(args.seed) 
    init_size = args.init_size
    print("init: ", args.init)
    print("init size: ", args.init_size)


    ### Initial annotated pool $L_0$ (random, clustering, or none)
    if args.init == 'random':
        init_ind = random.sample(range(len(kwargs['train_examples'])),init_size)
        pool_idx = list(range(len(kwargs['embeddings'])))
        for i in init_ind:
            pool_idx.remove(i)

        #naive clustering -- assign cluster centroids on random points
        cur_examples = [kwargs["train_examples"][idx] for idx in init_ind]
        _, clustering_model = cluster(embeddings=kwargs['embeddings'][init_ind],select_num=init_size, examples=cur_examples, thres=False, reverse=False)
        
    elif args.init == 'none':
        init_ind = []
        pool_idx = list(range(len(kwargs['embeddings'])))

    elif args.init == 'cluster':
        init_ind, clustering_model = cluster(embeddings=kwargs['embeddings'], examples = kwargs['train_examples'],select_num=init_size, thres=False, seed=args.seed)
        pool_idx = list(range(len(kwargs['embeddings'])))
        for i in init_ind:
            pool_idx.remove(i)

    print("Initial idxs: ",init_ind )

    

    if args.selective_annotation_method=='random':

        phase = 0
        selected_indices = random.sample(pool_idx,args.annotation_size)
        for i in selected_indices:
            pool_idx.remove(i)
        selected_indices += init_ind.copy()
 
    elif args.selective_annotation_method=='all':
        train_examples = kwargs['train_examples']
        selected_indices = range(len(train_examples))

    elif args.selective_annotation_method=='fast_votek':
        
        phase = 0
        selected_indices = fast_votek_mod(embeddings=kwargs['embeddings'], selected_indices=init_ind, select_num=args.annotation_size,k=150,
                                         vote_file=os.path.join(args.output_dir,'nearest_neighbors.json'))
        for i in selected_indices:
            pool_idx.remove(i)
        selected_indices += init_ind.copy()
        

    elif args.selective_annotation_method=='votek':

        phase = 1
        selected_indices = init_ind.copy()
        selected_indices_new = votek_mod(init_ind,
                                    pool_idx,
                                    train_embs=kwargs['embeddings'],
                                    test_embs=kwargs['embeddings'],
                                    train_examples=kwargs['train_examples'],
                                    test_examples=kwargs['train_examples'],
                                    return_string=kwargs['return_string'],
                                    format_example=kwargs['format_example'],
                                    maximum_input_len=kwargs['maximum_input_len'],
                                    label_map=kwargs['label_map'],
                                    single_context_example_len=kwargs['single_context_example_len'],
                                    inference_model=kwargs['inference_model'],
                                    inference_data_module=kwargs['inference_data_module'],
                                    tokenizer_gpt=kwargs['tokenizer_gpt'],
                                    args=args,
                                    k=150)
        selected_indices += selected_indices_new
        for idx in selected_indices_new:
            pool_idx.remove(idx)

    elif args.selective_annotation_method=='hardest':

        phase = 1
        selected_indices = init_ind.copy()

        uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                            pool_idx, 
                                            train_embs=kwargs['embeddings'],
                                            test_embs=kwargs['embeddings'],
                                            train_examples=kwargs['train_examples'],
                                            test_examples=kwargs['train_examples'],
                                            return_string=kwargs['return_string'],
                                            format_example=kwargs['format_example'],
                                            maximum_input_len=kwargs['maximum_input_len'],
                                            label_map=kwargs['label_map'],
                                            single_context_example_len=kwargs['single_context_example_len'],
                                            inference_model=kwargs['inference_model'],
                                            inference_data_module=kwargs['inference_data_module'],
                                            tokenizer_gpt=kwargs['tokenizer_gpt'],
                                            args=args,
                                            step=0)

        if args.evaluate_calibration:
            ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'])
            embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'])

            with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
                f.write(f"{len(kwargs['train_examples'])} train examples, accuracy is {acc}, ece is {ece_score}\n")    
            
        
        selected_indices_new = list(sorted_scores.keys())[:args.annotation_size]
        
        assert len(selected_indices_new) == args.annotation_size

        selected_indices += selected_indices_new
        for idx in selected_indices_new:
            pool_idx.remove(idx)


    elif args.selective_annotation_method=='adaicl_base':

        phase = 1
        selected_indices = init_ind.copy()

        hard_limit1 = 0
        hard_limit2 = args.hard_limit #1/2

        uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                            pool_idx, 
                                            train_embs=kwargs['embeddings'],
                                            test_embs=kwargs['embeddings'],
                                            train_examples=kwargs['train_examples'],
                                            test_examples=kwargs['train_examples'],
                                            return_string=kwargs['return_string'],
                                            format_example=kwargs['format_example'],
                                            maximum_input_len=kwargs['maximum_input_len'],
                                            label_map=kwargs['label_map'],
                                            single_context_example_len=kwargs['single_context_example_len'],
                                            inference_model=kwargs['inference_model'],
                                            inference_data_module=kwargs['inference_data_module'],
                                            tokenizer_gpt=kwargs['tokenizer_gpt'],
                                            args=args,
                                            step=0)

        if args.evaluate_calibration:
            ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'])
            embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'])

            with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
                f.write(f"{len(kwargs['train_examples'])} train examples, accuracy is {acc}, ece is {ece_score}\n")    
            
        hard_ind1 =  int(len(uncertainty_indices) * hard_limit1)
        hard_ind2 =  int(len(uncertainty_indices) * hard_limit2)
            
        hard_set_idx = list(sorted_scores.keys())[hard_ind1:hard_ind2]
        flag_idx = []
        new_list = []
        new_map = {}
        len_new = 0
        for idx in hard_set_idx:
            new_list.append(idx)
            new_map[len_new] = idx
            flag_idx.append(len_new)
            len_new += 1
        assert len(flag_idx) == len(hard_set_idx)

        

        cur_examples = [kwargs["train_examples"][idx] for idx in hard_set_idx]
        selected_indices_before, _ = cluster(embeddings=kwargs['embeddings'][hard_set_idx],select_num=args.annotation_size, examples=cur_examples, flag_idx = flag_idx, thres=True, reverse=False)

        print("before: ", selected_indices_before)
        selected_indices_new = []
        for idx in selected_indices_before:
            selected_indices_new.append(new_map[idx])


        print("New indx: ", selected_indices_new)
        assert len(selected_indices_new) == args.annotation_size

        selected_indices += selected_indices_new
        for idx in selected_indices_new:
            pool_idx.remove(idx)
        



    if args.selective_annotation_method != 'all':
        print("Selected total: ", len(selected_indices))
        assert  len(selected_indices) == args.annotation_size + init_size


    if args.evaluate_calibration:
        uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                            pool_idx, 
                                            train_embs=kwargs['embeddings'],
                                            test_embs=kwargs['embeddings'],
                                            train_examples=kwargs['train_examples'],
                                            test_examples=kwargs['train_examples'],
                                            return_string=kwargs['return_string'],
                                            format_example=kwargs['format_example'],
                                            maximum_input_len=kwargs['maximum_input_len'],
                                            label_map=kwargs['label_map'],
                                            single_context_example_len=kwargs['single_context_example_len'],
                                            inference_model=kwargs['inference_model'],
                                            inference_data_module=kwargs['inference_data_module'],
                                            tokenizer_gpt=kwargs['tokenizer_gpt'],
                                            args=args,
                                            step=phase)
        ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'], phase=phase)
        embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'], phase=phase)

        with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
            f.write(f"{len(pool_idx)} train examples, accuracy is {acc}, ece is {ece_score}\n")

    return selected_indices





def selective_annotation_adaptive_phases(args,**kwargs):

    """
    Multi-step annotation methods.
    AdaICL-base, AdaICL (default), AdaIC+ (default), AdaICL (with args)

    Returns:
        list: selected data points for annotation
    """

    random.seed(args.seed) 
    init_size = args.init_size
    print("init: ", args.init)
    print("init size: ", args.init_size)

    ### Initial annotated pool $L_0$ (random, clustering, or none)
    if args.init == 'random':
        init_ind = random.sample(range(len(kwargs['train_examples'])),init_size)
        pool_idx = list(range(len(kwargs['embeddings'])))
        for i in init_ind:
            pool_idx.remove(i)

        #naive clustering -- assign cluster centroids on random points
        cur_examples = [kwargs["train_examples"][idx] for idx in init_ind]
        _, clustering_model = cluster(embeddings=kwargs['embeddings'][init_ind],select_num=init_size, examples=cur_examples, thres=False, reverse=False)
        
    elif args.init == 'none':
        init_ind = []
        pool_idx = list(range(len(kwargs['embeddings'])))

    elif args.init == 'cluster':
        init_ind, clustering_model = cluster(embeddings=kwargs['embeddings'], examples = kwargs['train_examples'],select_num=init_size, thres=False)
        pool_idx = list(range(len(kwargs['embeddings'])))
        for i in init_ind:
            pool_idx.remove(i)

    print("Initial idxs: ",init_ind )

    if args.selective_annotation_method=='random':
        phase_size = args.annotation_size // args.phases
        selected_indices = init_ind

        for phase in range(args.phases):

            uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                                pool_idx, 
                                                train_embs=kwargs['embeddings'],
                                                test_embs=kwargs['embeddings'],
                                                train_examples=kwargs['train_examples'],
                                                test_examples=kwargs['train_examples'],
                                                return_string=kwargs['return_string'],
                                                format_example=kwargs['format_example'],
                                                maximum_input_len=kwargs['maximum_input_len'],
                                                label_map=kwargs['label_map'],
                                                single_context_example_len=kwargs['single_context_example_len'],
                                                inference_model=kwargs['inference_model'],
                                                inference_data_module=kwargs['inference_data_module'],
                                                tokenizer_gpt=kwargs['tokenizer_gpt'],
                                                args=args,
                                                step=phase)
            if args.evaluate_calibration:
                ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'], phase=phase)
                embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'], phase=phase)

                with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
                    f.write(f"{len(pool_idx)} train examples, accuracy is {acc}, ece is {ece_score}\n")    
            
            selected_indices_new = random.sample(pool_idx,phase_size)
            print("Selected new:", selected_indices_new)
            for i in selected_indices_new:
                pool_idx.remove(i)
            selected_indices += selected_indices_new
            

    if args.selective_annotation_method=='votek':

        annotation_size_original = args.annotation_size
        selected_indices = init_ind

        args.annotation_size = args.annotation_size // args.phases

        for phase in range(args.phases):
            new_selected_indices = votek_mod(selected_indices,
                                        pool_idx,
                                        train_embs=kwargs['embeddings'],
                                        test_embs=kwargs['embeddings'],
                                        train_examples=kwargs['train_examples'],
                                        test_examples=kwargs['train_examples'],
                                        return_string=kwargs['return_string'],
                                        format_example=kwargs['format_example'],
                                        maximum_input_len=kwargs['maximum_input_len'],
                                        label_map=kwargs['label_map'],
                                        single_context_example_len=kwargs['single_context_example_len'],
                                        inference_model=kwargs['inference_model'],
                                        inference_data_module=kwargs['inference_data_module'],
                                        tokenizer_gpt=kwargs['tokenizer_gpt'],
                                        args=args,
                                        k=15,
                                        step=phase)
            selected_indices += new_selected_indices
            for idx in new_selected_indices:
                pool_idx.remove(idx)

        args.annotation_size = annotation_size_original

    elif args.selective_annotation_method=='hardest':

        
        selected_indices = init_ind.copy()

        phase_size = args.annotation_size // args.phases

        for phase in range(args.phases):

            uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                                pool_idx, 
                                                train_embs=kwargs['embeddings'],
                                                test_embs=kwargs['embeddings'],
                                                train_examples=kwargs['train_examples'],
                                                test_examples=kwargs['train_examples'],
                                                return_string=kwargs['return_string'],
                                                format_example=kwargs['format_example'],
                                                maximum_input_len=kwargs['maximum_input_len'],
                                                label_map=kwargs['label_map'],
                                                single_context_example_len=kwargs['single_context_example_len'],
                                                inference_model=kwargs['inference_model'],
                                                inference_data_module=kwargs['inference_data_module'],
                                                tokenizer_gpt=kwargs['tokenizer_gpt'],
                                                args=args,
                                                step=phase)

            if args.evaluate_calibration:
                ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'], phase=phase)
                embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'], phase=phase)

                with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
                    f.write(f"{len(kwargs['train_examples'])} train examples, accuracy is {acc}, ece is {ece_score}\n")    
                
            
            selected_indices_new = list(sorted_scores.keys())[:phase_size]
            
            assert len(selected_indices_new) == phase_size

            selected_indices += selected_indices_new
            for idx in selected_indices_new:
                pool_idx.remove(idx)



    elif args.selective_annotation_method=='adaicl_base':

        selected_indices = init_ind.copy()

        hard_limit1 = 0
        hard_limit2 = 1/2

        phase_size = args.annotation_size // args.phases

        for phase in range(args.phases):

            uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                                pool_idx, 
                                                train_embs=kwargs['embeddings'],
                                                test_embs=kwargs['embeddings'],
                                                train_examples=kwargs['train_examples'],
                                                test_examples=kwargs['train_examples'],
                                                return_string=kwargs['return_string'],
                                                format_example=kwargs['format_example'],
                                                maximum_input_len=kwargs['maximum_input_len'],
                                                label_map=kwargs['label_map'],
                                                single_context_example_len=kwargs['single_context_example_len'],
                                                inference_model=kwargs['inference_model'],
                                                inference_data_module=kwargs['inference_data_module'],
                                                tokenizer_gpt=kwargs['tokenizer_gpt'],
                                                args=args,
                                                step=phase)

            if args.evaluate_calibration:
                ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'], phase=phase)
                embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'], phase=phase)

                with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
                    f.write(f"{len(kwargs['train_examples'])} train examples, accuracy is {acc}, ece is {ece_score}\n")    
                
            hard_ind1 =  int(len(uncertainty_indices) * hard_limit1)
            hard_ind2 =  int(len(uncertainty_indices) * hard_limit2)
                
            hard_set_idx = list(sorted_scores.keys())[hard_ind1:hard_ind2]
            flag_idx = []
            new_list = []
            new_map = {}
            len_new = 0
            for idx in range(len(kwargs["train_examples"])):
                if idx in uncertainty_indices:
                    new_list.append(idx)
                    new_map[len_new] = idx
                    if idx in hard_set_idx:
                        flag_idx.append(len_new)
                    len_new += 1
            assert len(flag_idx) == len(hard_set_idx)

            

            cur_examples = [kwargs["train_examples"][idx] for idx in hard_set_idx]
            selected_indices_before, _ = cluster(embeddings=kwargs['embeddings'][hard_set_idx],select_num=phase_size, examples=cur_examples, flag_idx = flag_idx, thres=True, reverse=False)

            print("before: ", selected_indices_before)
            selected_indices_new = []
            for idx in selected_indices_before:
                selected_indices_new.append(new_map[idx])
            
            assert len(selected_indices_new) == phase_size

            selected_indices += selected_indices_new
            for idx in selected_indices_new:
                pool_idx.remove(idx)


    elif args.selective_annotation_method=='ada_icl_plus_default':

        selected_indices = init_ind.copy()

        hard_limit1 = 0
        hard_limit2 = 1/2


        phase_size = args.annotation_size // args.phases

        with open(os.path.join(args.output_dir, f"selected_indices_0.json"),'w') as f:
            json.dump(selected_indices,f,indent=4)

        for phase in range(args.phases):

            uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                                pool_idx, 
                                                train_embs=kwargs['embeddings'],
                                                test_embs=kwargs['embeddings'],
                                                train_examples=kwargs['train_examples'],
                                                test_examples=kwargs['train_examples'],
                                                return_string=kwargs['return_string'],
                                                format_example=kwargs['format_example'],
                                                maximum_input_len=kwargs['maximum_input_len'],
                                                label_map=kwargs['label_map'],
                                                single_context_example_len=kwargs['single_context_example_len'],
                                                inference_model=kwargs['inference_model'],
                                                inference_data_module=kwargs['inference_data_module'],
                                                tokenizer_gpt=kwargs['tokenizer_gpt'],
                                                args=args,
                                                step=phase)

            if args.evaluate_calibration:
                ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'], phase=phase)
                embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'], phase=phase)

                with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
                    f.write(f"{len(kwargs['train_examples'])} train examples, accuracy is {acc}, ece is {ece_score}\n")    
                
            ind1 =  int(len(uncertainty_indices) * hard_limit1)
            ind2 =  int(len(uncertainty_indices) * hard_limit2)
                
            flag_idx = list(sorted_scores.keys())[ind1:ind2]
            easy_idx = list(sorted_scores.keys())[ind2:]


            k_graph = 15
            two_hop = False

            print("kgraph and two hop: ", k_graph, two_hop)
            selected_indices_new = density_max_coverage(embeddings=kwargs['embeddings'], 
                                                           hard_idx = flag_idx.copy(), 
                                                            easy_idx = easy_idx,
                                                           selected_indices=selected_indices, 
                                                           select_num=phase_size,
                                                           k=k_graph,
                                                           two_hop=two_hop,
                                                           weighted = True,
                                                           thres_graph=False,
                                                           mc_selection="hard",
                                         vote_file=os.path.join(args.output_dir,'nearest_neighbors.json'))

            selected_indices += selected_indices_new

            with open(os.path.join(args.output_dir, f"selected_indices_{phase+1}.json"),'w') as f:
                json.dump(selected_indices,f,indent=4)
            for idx in selected_indices_new:
                pool_idx.remove(idx)

    
    elif args.selective_annotation_method=='ada_icl_default':

        selected_indices = init_ind.copy()

        hard_limit1 = 0
        hard_limit2 = 1/2


        phase_size = args.annotation_size // args.phases


        phase = 0
        cur_annotated_size = 0

        with open(os.path.join(args.output_dir, f"selected_indices_0.json"),'w') as f:
            json.dump(selected_indices,f,indent=4)

        while cur_annotated_size < args.annotation_size:
        

            uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                                pool_idx, 
                                                train_embs=kwargs['embeddings'],
                                                test_embs=kwargs['embeddings'],
                                                train_examples=kwargs['train_examples'],
                                                test_examples=kwargs['train_examples'],
                                                return_string=kwargs['return_string'],
                                                format_example=kwargs['format_example'],
                                                maximum_input_len=kwargs['maximum_input_len'],
                                                label_map=kwargs['label_map'],
                                                single_context_example_len=kwargs['single_context_example_len'],
                                                inference_model=kwargs['inference_model'],
                                                inference_data_module=kwargs['inference_data_module'],
                                                tokenizer_gpt=kwargs['tokenizer_gpt'],
                                                args=args,
                                                step=phase)
            
            if args.evaluate_calibration:
                ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'], phase=phase)
                embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'], phase=phase)

                with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
                    f.write(f"{len(kwargs['train_examples'])} train examples, accuracy is {acc}, ece is {ece_score}\n")    

            
            ind1 =  int(len(uncertainty_indices) * hard_limit1)
            ind2 =  int(len(uncertainty_indices) * hard_limit2)
                
            flag_idx = list(sorted_scores.keys())[ind1:ind2]
            easy_idx = list(sorted_scores.keys())[ind2:]
            


            k_graph = 5
            two_hop = True
            print("kgraph and two hop: ", k_graph, two_hop)
            selected_indices_new = density_max_coverage(embeddings=kwargs['embeddings'], 
                                                           hard_idx = flag_idx.copy(), 
                                                           easy_idx = easy_idx,
                                                           selected_indices=selected_indices, 
                                                           select_num= args.annotation_size - cur_annotated_size,
                                                           k=k_graph,
                                                           two_hop=two_hop,
                                                           weighted = False,
                                                           thres_graph=False,
                                                           mc_selection="hard",
                                         vote_file=os.path.join(args.output_dir,'nearest_neighbors.json'))

            print("Len selected_indices_new at phase:", phase, len(selected_indices_new))

            selected_indices += selected_indices_new
            cur_annotated_size += len(selected_indices_new)

            with open(os.path.join(args.output_dir, f"selected_indices_{phase+1}.json"),'w') as f:
                json.dump(selected_indices,f,indent=4)
            for idx in selected_indices_new:
                if idx in pool_idx:
                    pool_idx.remove(idx)
            phase += 1
        args.phases = phase


    elif args.selective_annotation_method=='ada_icl':

        selected_indices = init_ind.copy()

        hard_limit = args.hard_limit
        assert hard_limit < 1 and hard_limit > 0


        phase_size = args.annotation_size // args.phases
        phase = 0
        cur_annotated_size = 0


        with open(os.path.join(args.output_dir, f"selected_indices_0.json"),'w') as f:
            json.dump(selected_indices,f,indent=4)

        while cur_annotated_size < args.annotation_size:
        

            uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                                pool_idx, 
                                                train_embs=kwargs['embeddings'],
                                                test_embs=kwargs['embeddings'],
                                                train_examples=kwargs['train_examples'],
                                                test_examples=kwargs['train_examples'],
                                                return_string=kwargs['return_string'],
                                                format_example=kwargs['format_example'],
                                                maximum_input_len=kwargs['maximum_input_len'],
                                                label_map=kwargs['label_map'],
                                                single_context_example_len=kwargs['single_context_example_len'],
                                                inference_model=kwargs['inference_model'],
                                                inference_data_module=kwargs['inference_data_module'],
                                                tokenizer_gpt=kwargs['tokenizer_gpt'],
                                                args=args,
                                                step=phase)
            
            if args.evaluate_calibration:
                ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'], phase=phase)
                embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'], phase=phase)

                with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
                    f.write(f"{len(kwargs['train_examples'])} train examples, accuracy is {acc}, ece is {ece_score}\n")    

            
            ind_hard =  int(len(uncertainty_indices) * hard_limit)
                
            flag_idx = list(sorted_scores.keys())[:ind_hard]
            easy_idx = list(sorted_scores.keys())[ind_hard:]
            

            k_graph = args.k_graph
            two_hop = args.two_hop
            weighted = args.ada_icl_plus
            if args.ada_icl_plus:
                select_num_mcp = phase_size
            else:
                select_num_mcp = args.annotation_size - cur_annotated_size
            print("kgraph and two hop: ", k_graph, two_hop)
            selected_indices_new = density_max_coverage(embeddings=kwargs['embeddings'], 
                                                           hard_idx = flag_idx.copy(), 
                                                           easy_idx = easy_idx,
                                                           selected_indices=selected_indices, 
                                                           select_num= select_num_mcp,
                                                           k=k_graph,
                                                           two_hop=two_hop,
                                                           weighted = weighted,
                                                           thres_graph=args.thres_graph,
                                                           mc_selection=args.mc_selection,
                                         vote_file=os.path.join(args.output_dir,'nearest_neighbors.json'))

            print("Len selected_indices_new at phase:", phase, len(selected_indices_new))

            selected_indices += selected_indices_new
            cur_annotated_size += len(selected_indices_new)

            with open(os.path.join(args.output_dir, f"selected_indices_{phase+1}.json"),'w') as f:
                json.dump(selected_indices,f,indent=4)
            for idx in selected_indices_new:
                if idx in pool_idx:
                    pool_idx.remove(idx)
            phase += 1
        args.phases = phase

    if len(selected_indices) < args.annotation_size + init_size:
        rest_num = args.annotation_size + init_size - len(selected_indices)
        selected_indices += random.sample(pool_idx, rest_num)


    if args.selective_annotation_method != 'all':
        print("Selected total: ", len(selected_indices))
        assert  len(selected_indices) == args.annotation_size + init_size


    if args.evaluate_calibration:
        uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                            pool_idx, 
                                            train_embs=kwargs['embeddings'],
                                            test_embs=kwargs['embeddings'],
                                            train_examples=kwargs['train_examples'],
                                            test_examples=kwargs['train_examples'],
                                            return_string=kwargs['return_string'],
                                            format_example=kwargs['format_example'],
                                            maximum_input_len=kwargs['maximum_input_len'],
                                            label_map=kwargs['label_map'],
                                            single_context_example_len=kwargs['single_context_example_len'],
                                            inference_model=kwargs['inference_model'],
                                            inference_data_module=kwargs['inference_data_module'],
                                            tokenizer_gpt=kwargs['tokenizer_gpt'],
                                            args=args,
                                            step=args.phases)
        ece_score, acc = reliability_plot(args, kwargs['label_map'], kwargs['train_examples'], phase=args.phases)
        embedding_plot(args,kwargs['label_map'],selected_indices,kwargs['embeddings'], phase=args.phases)

        with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
            f.write(f"{len(kwargs['train_examples'])} train examples, accuracy is {acc}, ece is {ece_score}\n")
        


    return selected_indices

