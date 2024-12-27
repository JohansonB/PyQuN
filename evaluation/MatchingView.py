
class MatchingView:
    @staticmethod
    def display(res: 'ExperimentResult') -> None:
        """
        Displays a matching where each match is displayed in a table format.
        Attributes are ordered by their occurrence count: most common attributes
        appear first, and less-common attributes are shifted to the right.
        """
        matching = res.match
        for match in matching.get_matches():
            print(f"Match ID: {match.get_id()}")


            attribute_counts = {}
            all_keys = set()
            for element in match:
                for attr in element:
                    key = attr.value
                    all_keys.add(key)
                    attribute_counts[key] = attribute_counts.get(key, 0) + 1


            sorted_keys = sorted(all_keys, key=lambda k: (-attribute_counts[k], k))


            header = ["Element ID", "Name"] + sorted_keys


            col_widths = [len(col) for col in header]
            rows = []


            for element in match:
                row = [str(element.get_id()), element.get_name() or ""]
                for key in sorted_keys:
                    value = next((str(attr.value) for attr in element if attr.value == key), "")
                    row.append(value)
                rows.append(row)
                col_widths = [max(col_widths[i], len(row[i])) for i in range(len(row))]

            formatted_header = " | ".join(
                f"{header[i]:<{col_widths[i]}}" for i in range(len(header))
            )
            print(formatted_header)
            print("-" * len(formatted_header))

            for row in rows:
                formatted_row = " | ".join(
                    f"{row[i]:<{col_widths[i]}}" for i in range(len(row))
                )
                print(formatted_row)

            print("\n")
