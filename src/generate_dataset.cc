#include <iostream>
#include <algorithm>
#include <string>
#include "dsl/utils.h"
#include "dataset-generator.h"
#include "attribute.h"

using namespace std;
using namespace dsl;

void output_value(const Value &value) {
    if (OptExists(value.integer())) {
        cout << OptValue(value.integer());
    } else {
        auto l = OptValue(value.list());
        cout << "[";
        for (auto i = 0; i < l.size(); i++) {
            auto x = l.at(i);

            cout << x;
            if (i != (l.size() - 1)) {
                cout << ",";
            }
        }
        cout << "]";
    }
}
void output_input(const Input &input) {
    cout << "[";
    for (auto i = 0; i < input.size(); i++) {
        auto x = input.at(i);

        output_value(x);
        if (i != (input.size() - 1)) {
            cout << ",";
        }
    }
    cout << "]";
}

void output_attribute(const Attribute &attr) {
    std::vector<double> vec = attr;
    cout << "[";
    for (auto i = 0; i < vec.size(); i++) {
        cout << vec.at(i);
        if (i != (vec.size() - 1)) {
            cout << ",";
        }
    }
    cout << "]";
}
void func(const Argument& argument, const TypeEnvironment& env)
{
	auto it = env.find(OptValue(argument.variable()));
	if (it == env.end())
	{
		std::cout << "NO" << endl;
	}
	else
		std::cout << "Yes" << endl;
}



int main(int argc, char **argv) {

	///Test
	//TypeEnvironment env;
	//Variable var = { 0 };
	//env.insert({ var,dsl::Type::Integer });

	//Argument arg(var);

	//func(arg, env);

	///Test end



    size_t max_length = 4;
    size_t dataset_size = 0;
    size_t example_pair_per_program = 1;
	 
    if (argc >= 2) {
        max_length = atoi(argv[1]);
    }
    if (argc >= 3) {
        dataset_size = atoi(argv[2]);
    }
    if (argc >= 4) {
        example_pair_per_program = atoi(argv[3]);
    }

    cerr << "Generate dataset\n" << "  Max-Length: " << max_length << "\n  Dataset-Size: " << dataset_size << endl;
	auto time_start = std::chrono::system_clock::now();
    auto dataset = generate_dataset(1, max_length, dataset_size, example_pair_per_program * EXAMPLE_NUM);
	auto time_end = std::chrono::system_clock::now();
	std::cerr <<"Cost "<< std::chrono::duration<double>(time_end - time_start).count() << "s\n";
    cout << "[\n";
	long long int cnt = 0;
    if (OptExists(dataset)) {
        auto x = OptValue(dataset);
		sort(x.programs.begin(), x.programs.end(), [](const auto  &a_pair, const auto & b_pair) {
		
			const auto & a = a_pair.first;
			const auto & b = b_pair.first;
			if (a.size() != b.size())
				return a.size() < b.size();
			for (int i = 0; i < a.size(); i++)
			{
				if (a[i].function != b[i].function)
					return a[i].function < b[i].function;
			}

			for (int i = 0; i < a.size(); i++)
			{
				if (a[i].arguments.size() != b[i].arguments.size())
					return a[i].arguments.size() < b[i].arguments.size();
				for (int j = 0; j < a[i].arguments.size(); j++)
				{
					if (a[i].arguments[j].Val() != b[i].arguments[j].Val())
						return a[i].arguments[j].Val() < b[i].arguments[j].Val();
				 }

			}
		});
        for (const auto &p: x.programs) {
            cnt += 1;
            const auto &program = p.first;
            const auto &examples = p.second;
            auto attribute = Attribute(program);
            cerr << "# Program\n" << program << flush;
            auto pair_num = examples.size() / EXAMPLE_NUM;
            for (auto j = 0; j < pair_num; ++j) {
                cout << "{\"examples\":[\n";
                for (auto k = 0; k < EXAMPLE_NUM; ++k) {
                    const auto &example = examples.at(j * EXAMPLE_NUM + k);

                    cout << "{\"input\":";
                    output_input(example.input);
                    cout << ",\"output\":";
                    output_value(example.output);
                    cout << "}";
                    if (k != EXAMPLE_NUM - 1) {
                        cout << ",";
                    }
                    cout << "\n";

                }
                cout << "],\n\"attribute\":";
                output_attribute(attribute);

                cout << "}";
                if (cnt != x.programs.size() ||
                    j != pair_num - 1) {
                    cout << ",";
                }
                cout << "\n" << flush;
            }
        }
    } else {
        cerr << "Fail to generate dataset" << endl;
    }
    cout << "]" << endl;
	cerr << "Total# " << cnt << endl;
    return 0;
}