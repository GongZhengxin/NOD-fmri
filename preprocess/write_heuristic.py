class heucomponent:
    def __init__(self, file):
        self.file = open(file, 'a+')
        self.HEADER = ['import os\ndef create_key(template, outtype=(\'nii.gz\',), annotation_classes=None):\n    if template is None or not template:\n        raise ValueError(\'Template must be a valid format string\')\n    return template, outtype, annotation_classes\ndef infotodict(seqinfo):\n    """Heuristic evaluator for determining which runs belong where\n    allowed template fields - follow python string module:\n    item: index within category\n    subject: participant id\n    seqitem: run number during scanning\n    subindex: sub index within group\n    """\n\n']
        
    def write_catalog(self,task_list):
        """
        write catalog rules part
        parameters:
        -----------
        task_list : list, value of session_task dict, like ['func_rest', 'fmap/magnitude']
        """
        content = []
        for _ in task_list:
            mod, label = _.split('/')[0], _.split('/')[1]
            if mod in ['anat', 'dwi', 'fmap']:
                content.append("    {0}_{1} = create_key('sub-{{subject}}/{{session}}/{0}/sub-{{subject}}_{{session}}_run-{{item:02d}}_{1}')\n"\
                    .format(mod, label))
            if mod in ['func']:
                content.append("    {0}_{1} = create_key('sub-{{subject}}/{{session}}/{0}/sub-{{subject}}_{{session}}_task-{1}_run-{{item:02d}}_bold')\n"\
                    .format(mod, label))
        self.file.writelines(content)

    def write_info(self, task_list):
        """
        write the info dict part
        parameters:
        -----------
        task_list: list, value of session_task dict, like ['func_rest', 'fmap/magnitude']
        """
        
        content = ["\n    info = {"] + ["{0[0]}_{0[1]}:[],".format(_.split('/')) for _ in task_list[:-1]] \
            + ["{0[0]}_{0[1]}:[]}}\n".format(_.split('/')) for _ in [task_list[-1]]]
        
        self.file.writelines(content)

    def write_condition(self, task_list, feature_dict):
        """
        write the condition part
        parameters:
        ----------
        task_list: list
        feaure_dict: dict
        """
        openning = ["\n    for idx, s in enumerate(seqinfo):\n"]
        ending = ["    return info\n"]
        middle = []
        for _ in task_list:
            mod, label = _.split('/')[0], _.split('/')[1]
            if mod == 'anat':
                middle.append("        if ('{}' in s.protocol_name):\n".format(feature_dict[_][0]))
                middle.append("            info[{0}_{1}].append(s.series_id)\n".format(mod, label))
            if mod == 'fmap':
                middle.append("        if ('{0[0]}' in s.protocol_name) and (s.dim3 == {0[1]}):\n".format(feature_dict[_]))
                middle.append("            info[{0}_{1}].append(s.series_id)\n".format(mod, label))
            if mod == 'func':
                middle.append("        if ('{0[0]}' in s.protocol_name) and (s.dim4 == {0[1]}):\n".format(feature_dict[_]))
                middle.append("            info[{0}_{1}].append(s.series_id)\n".format(mod, label))
        content = openning + middle + ending
        self.file.writelines(content)

    def create_heristic(self, task_list, feature_dict):
        """
        create the heuristic.py according to task_list & feature_dict
        parameters:
        -----------
        task_list: list
        feature_dict: dict
        """
        self.file.writelines(self.HEADER)
        self.write_catalog(task_list)
        self.write_info(task_list)
        self.write_condition(task_list, feature_dict)
        self.file.close()

