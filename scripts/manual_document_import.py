#!/usr/bin/env python
import os
import sys
import uuid
import django
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'auth.settings')
django.setup()

from chatbot.models import Document, DocumentChunk

# First, delete any existing documents with this title
Document.objects.filter(title__contains="Décret-loi n° 2011-88").delete()
print("Deleted any existing documents")

# Create the document with full content
doc_title = "Décret-loi n° 2011-88 du 24 septembre 2011"
document = Document.objects.create(
    title=doc_title,
    content="Loi sur les associations en Tunisie",  # Short placeholder
    language='fr'
)
print(f"Created document: {document.title}")


# Define a simple fixed-size chunking function that won't get into loops
def create_fixed_chunks(text, chunk_size=250):
    """Create chunks of fixed size, breaking at spaces when possible"""
    chunks_created = 0
    position = 0
    text_length = len(text)

    # Safe guard to prevent infinite loops
    max_chunks = (text_length // 100) + 10  # Allow some extra chunks just in case

    print(f"Creating chunks from text of length {text_length}")

    while position < text_length and chunks_created < max_chunks:
        # Determine end position for this chunk
        end_pos = min(position + chunk_size, text_length)

        # If we're not at the end of text, try to break at a space
        if end_pos < text_length:
            # Look for the last space within the chunk limit
            last_space = text.rfind(' ', position, end_pos)
            if last_space > position:
                end_pos = last_space + 1  # Include the space

        # Extract the chunk
        chunk = text[position:end_pos]

        # Create a unique ID for this chunk
        chunk_id = f"{document.id}_{uuid.uuid4()}"

        # Store the chunk in the database
        DocumentChunk.objects.create(
            document=document,
            content=chunk,
            chunk_id=chunk_id
        )

        # Move position for next chunk
        position = end_pos
        chunks_created += 1

        # Show progress every 10 chunks
        if chunks_created % 10 == 0:
            print(f"Created {chunks_created} chunks, processed {position}/{text_length} characters")

    print(f"Total chunks created: {chunks_created}")
    return chunks_created


# The complete document text (from your paste)
full_text =[
    # Section 1: Introduction et principes généraux
    """Décret-loi n° 2011-88 du 24 septembre 2011, portant organisation des associations.
Le Président de la République par intérim, 
Sur proposition de la haute instance pour la réalisation des objectifs de la révolution, de la réforme politique et de la transition démocratique,
Vu la loi organique n° 93-80 du 26 juillet 1993, relative à l'installation des organisations non gouvernementales en Tunisie,
Vu la loi n° 59-154 du 7 novembre 1959, relative aux associations,
Vu la loi n° 68-8 du 8 mars 1968, portant organisation de la cour des comptes, ensemble les textes qui l'ont modifié ou complété,
Vu le décret-loi n° 2011-6 du 18 février 2011, portant création de la haute instance pour la réalisation des objectifs de la révolution, de la réforme politique et de la transition démocratique,
Vu le décret-loi n° 2011-14 du 23 mars 2011, portant organisation provisoire des pouvoirs publics,
Vu le décret n° 70-118 du 11 avril 1970, portant organisation des services du Premier ministère, ensemble les textes qui l'ont modifié ou complété,
Vu la délibération du conseil des ministres,
Prend le décret-loi dont la teneur suit :

Chapitre premier
Principes Généraux

Article premier - Le présent décret-loi garantit la liberté de constituer des associations, d'y adhérer, d'y exercer des activités et le renforcement du rôle des organisations de la société civile ainsi que leur développement et le respect de leur indépendance.

Art. 2 - L'association est une convention par laquelle deux ou plusieurs personnes œuvrent d'une façon permanente, à réaliser des objectifs autres que la réalisation de bénéfices.

Art. 3 - Dans le cadre de leurs statuts, activités et financement, les associations respectent les principes de l'Etat de droit, de la démocratie, de la pluralité, de la transparence, de l'égalité et des droits de l'Homme tels que définis par les conventions internationales ratifiées par la République Tunisienne.

Art. 4 - Il est interdit à l'association :
Premièrement : de s'appuyer dans ses statuts ou communiqués ou programmes ou activités sur l'incitation à la violence, la haine, l'intolérance et la discrimination fondée sur la religion, le sexe ou la région.
Deuxièmement : d'exercer des activités commerciales en vue de distribuer des fonds au profit de ses membres dans leur intérêt personnel ou d'être utilisée dans le but d'évasion fiscale.
Troisièmement : de collecter des fonds en vue de soutenir des partis politiques ou des candidats indépendants à des élections nationales, régionales, locales ou leur procurer une aide matérielle. Cette interdiction n'inclut pas le droit de l'association à exprimer ses opinions politiques et ses positions par rapport aux affaires d'opinion publique.

Art. 5 - L'association a le droit :
Premièrement : d'obtenir des informations.
Deuxièmement : d'évaluer le rôle des institutions de l'Etat et de formuler des propositions en vue d'améliorer leur rendement.
Troisièmement : d'organiser des réunions, manifestations, congrès, ateliers de travail et toute autre activité civile.
Quatrièmement : de publier les rapports et les informations, éditer des publications et procéder aux sondages d'opinions.

Art. 6 - Il est interdit aux autorités publiques d'entraver ou de ralentir l'activité des associations de manière directe ou indirecte.

Art. 7 - l'Etat prend toutes les mesures nécessaires garantissant à tout individu sa protection par les autorités compétentes contre toute violence, menace, vengeance, discrimination préjudiciable de fait ou de droit, pression ou toute autre mesure abusive suite à l'exercice légitime de ses droits prévus par le présent décret-loi.""",

    # Section 2: Constitution des associations (Très important)
    """Chapitre II
La constitution des associations et leur gestion

Art. 8 - Premièrement : Toute personne physique, tunisienne ou étrangère résidente en Tunisie, a le droit de constituer une association ou d'y adhérer ou de s'en retirer conformément aux dispositions du présent décret-loi.
Deuxièmement : La personne physique fondatrice ne doit pas avoir moins de seize (16) ans.

Art. 9 - Les fondateurs et dirigeants de l'association ne peuvent pas être en charge de responsabilités au sein des organes centraux dirigeant les partis politiques.

Art. 10 - Premièrement : la constitution des associations est régie par le régime de déclaration.
Deuxièmement : les personnes désirant constituer une association doivent adresser au secrétaire général du gouvernement une lettre recommandée avec accusé de réception comportant :
a- Une déclaration indiquant la dénomination de l'association, son objet, ses objectifs, son siège et les sièges de ses filiales s'ils existent.
b- * Une copie de la carte d'identité nationale des personnes physiques tunisiennes fondatrices de l'association et le cas échéant, une copie de la carte d'identité du tuteur.
* Une copie de la carte de séjour pour les étrangers.
c- Les statuts en deux exemplaires signés par les fondateurs ou leurs représentants. Les statuts doivent comprendre les mentions suivantes :
1- la dénomination officielle de l'association en langue arabe et le cas échéant, en langue étrangère.
2- l'adresse du siège principal de l'association.
3- une présentation des objectifs de l'association ainsi que les moyens de leur réalisation.
4- les conditions d'adhésion, les cas de son extinction, ainsi que les droits et les obligations des membres.
5- la présentation de l'organigramme de l'association, le mode d'élection retenu et les prérogatives de chacun de ses organes.
6- la détermination de l'organe qui détient au sein de l'association, la prérogative de modification du règlement intérieur et de prise de décision concernant la dissolution, la fusion ou la scission.
7- la détermination des modes de prise de décisions et de règlement des différends.
8- le montant de la cotisation mensuelle ou annuelle s'il en existe.
Troisièmement : Un huissier de justice vérifie, lors de l'envoi de la lettre, l'existence des données susvisées, et en dresse un procès-verbal en deux exemplaires qu'il remet au représentant de l'association.

Art. 11 – Premièrement : Lors de la réception de l'accusé de réception, le représentant de l'association dépose dans un délai n'excédant pas sept (7) jours, une annonce à l'Imprimerie Officielle de la République Tunisienne indiquant la dénomination de l'association, son objet, ses objectifs, et son siège, accompagnée d'un exemplaire du procès-verbal susmentionné.
L'Imprimerie Officielle de la République Tunisienne publie impérativement l'annonce au Journal Officiel dans un délai de quinze (15) jours à compter du jour de son dépôt.
Deuxièmement : Le non-retour de l'accusé de réception dans les trente (30) jours suivant l'envoi de la lettre susvisée vaut réception.""",

    # Section 3: Personnalité juridique et droits
    """Art. 12 - L'association est réputée légalement constituée à compter du jour de l'envoi de la lettre mentionnée à l'article dix (10) et acquiert la personnalité morale à partir de la date de publication de l'annonce au Journal Officiel de la République Tunisienne.

Art. 13 - Les associations légalement constituées ont le droit d'ester en justice, d'acquérir, de posséder et d'administrer leurs ressources et biens. L'association peut également accepter les aides, dons, donations et legs.

Art. 14 - Toute association a le droit de se constituer partie civile ou d'intenter une action se rapportant à des actes relevant de son objet et ses objectifs prévus par ses statuts.
Néanmoins, si les actes sont commis contre des personnes déterminées, l'association ne peut intenter cette action que si elle en est mandatée par ces derniers et ce, par écrit explicite.

Art. 15 - Les fondateurs, dirigeants, salariés et adhérents à l'association ne sont pas tenus personnellement des obligations légales de l'association. Les créanciers de l'association ne peuvent pas leur réclamer le remboursement des créances à partir de leurs biens propres.

Art. 16 - Les dirigeants de l'association informent le secrétaire général du gouvernement, par lettre recommandée avec accusé de réception de toute modification apportée aux statuts de l'association dans un délai maximum d'un mois à compter de la prise de décision de modification. La modification est communiquée au public à travers les médias écrits et sur le site électronique de l'association s'il en existe.

Art. 17 - Sans préjudice des dispositions du présent décret loi, l'association fixe ses propres conditions d'adhésion. Le membre de l'association doit :
Premièrement : Etre de nationalité tunisienne ou être résident en Tunisie.
Deuxièmement : Avoir treize (13) ans.
Troisièmement : Accepter par écrit les statuts de l'association.
Quatrièmement : Verser le montant de cotisation à l'association.

Art. 18 - Les membres d'une association et ses salariés ne peuvent participer à l'élaboration ou la prise de décisions pouvant entraîner un conflit entre leurs intérêts personnels ou fonctionnels et ceux de l'association.

Art. 19 - Premièrement : Les statuts de l'association fixent impérativement les modalités de suspension provisoire de son activité ou de sa dissolution.
Deuxièmement : Les statuts de l'association fixent les règles de liquidation de ses biens et des fonds lui appartenant en cas de dissolution volontaire prévue par ses statuts.""",

    # Section 4: Associations étrangères
    """Chapitre III
Les associations étrangères

Art. 20 - Est réputée association étrangère toute filiale d'une association constituée conformément à la législation d'un autre Etat. La filiale de l'association étrangère en Tunisie est constituée conformément aux dispositions du présent décret loi.

Art. 21 – Premièrement : Le représentant de l'association étrangère adresse au secrétaire général du gouvernement une lettre recommandée avec accusé de réception comportant :
1- la dénomination de l'association.
2- l'adresse du siège principal de la filiale de l'association en Tunisie.
3- une présentation des activités que la filiale de l'association désire exercer en Tunisie.
4- les noms et adresses des dirigeants tunisiens ou étrangers résidents en Tunisie de la filiale de l'association étrangère.
5- une copie de la carte d'identité des dirigeants tunisiens et une copie de la carte de séjour ou du passeport des dirigeants étrangers.
6- deux exemplaires des statuts signés par les fondateurs ou leurs représentants.
7- un document officiel prouvant que l'association mère est légalement constituée à son pays d'origine.
Deuxièmement : Les informations et pièces mentionnées au paragraphe premier de cet article doivent être traduites en langue arabe par un interprète assermenté.
Troisièmement : Un huissier de justice vérifie lors de l'envoi de la lettre, l'existence des données susvisées et en dresse un procès verbal en deux exemplaires qu'il transmet au représentant de l'association.

Art. 22 – Premièrement : En cas de contradiction manifeste entre les statuts de l'association étrangère et les dispositions des articles 3 et 4 du présent décret loi, le secrétaire général du gouvernement peut, par décision motivée, refuser d'inscrire l'association, et ce, dans un délai de trente (30) jours à compter de la date de réception de la lettre mentionnée au paragraphe premier de l'article 21.
Les dirigeants de la filiale de l'association étrangère en Tunisie peuvent contester la légalité de la décision de refus d'inscription et ce conformément aux procédures en vigueur en matière d'excès de pouvoir conformément à la loi n° 72-40 du 1er juin 1972 relative au tribunal administratif.

Art. 24 - L'association étrangère peut constituer des filiales en Tunisie conformément aux dispositions du présent décret-loi.

Art. 25 – A l'exception des dispositions du présent chapitre, les associations étrangères sont soumises au même régime que les associations nationales.""",

    # Section 5: Réseaux et dissolution
    """Chapitre IV
Le réseau d'associations

Art. 26 - Deux ou plusieurs associations peuvent constituer un réseau d'associations.

Art. 27 - Le représentant du réseau adresse au secrétaire général du gouvernement une lettre recommandée avec accusé de réception comportant :
1- la déclaration de constitution.
2- les statuts du réseau.
3- une copie de l'annonce de constitution des associations formant le réseau.

Art. 29 - Le réseau acquiert une personnalité morale distincte de celles des associations qui le forment.

Art. 30 - Le réseau peut accepter l'adhésion de filiales d'associations étrangères.

Art. 31 – A l'exception des dispositions du présent chapitre, le réseau est soumis au même régime applicable aux associations nationales.

Chapitre V
Fusion et Dissolution

Art. 32 – Premièrement : Les associations ayant des objectifs similaires ou rapprochés peuvent fusionner et former une seule association, et ce, conformément aux statuts de chacune d'entre elles.
Deuxièmement : Les procédures de fusion et de constitution de la nouvelle association sont prévues par les dispositions du présent décret-loi.

Art. 33 – Premièrement : La dissolution de l'association est soit volontaire par décision de ses membres conformément aux statuts, soit judiciaire en vertu d'un jugement du tribunal.
Deuxièmement : Si l'association prend la décision de dissolution, elle est tenue d'en informer le secrétaire général du gouvernement par lettre recommandée avec accusé de réception, et ce, dans les trente (30) jours qui suivent la date de prise de décision de dissolution, et de désigner un liquidateur judiciaire.
Troisièmement : En cas de dissolution judiciaire, le tribunal procède à la désignation d'un liquidateur.
Quatrièmement : Pour répondre aux exigences de la liquidation, l'association présente un état de ses biens mobiliers et immobiliers qui sera retenu pour s'acquitter de ses obligations. Le reliquat sera distribué conformément aux statuts de l'association sauf si ces biens proviennent d'aides, dons, donations et legs. Dans ce cas, ils seront attribués à une autre association ayant des objectifs similaires et désignée par l'organe compétent de l'association.""",

    # Section 6: Dispositions financières et comptables
    """Chapitre VI
Dispositions financières

Art. 34 - Les ressources d'une association se composent des :
1- cotisations de ses membres,
2- aides publiques,
3- dons, donations et legs d'origine nationale ou étrangère,
4- recettes résultant de ses biens, activités et projets.

Art. 35 - Il est interdit aux associations d'accepter des aides, dons ou donations émanant d'Etats n'ayant pas de relations diplomatiques avec la Tunisie ou d'organisations défendant les intérêts et les politiques de ces Etats.

Art. 36 - L'Etat doit affecter les fonds nécessaires du budget à l'appui et au soutien des associations et ce, sur la base de la compétence, des projets et des activités. Les critères du financement public sont fixés par décret.

Art. 37 - Premièrement : l'association est tenue de consacrer ses ressources aux activités nécessaires à la réalisation de ses objectifs.
Deuxièmement : l'association peut participer aux appels d'offres annoncés par les autorités publiques, à condition que les matériaux ou les services requis dans l'appel d'offre relèvent de son activité.
Troisièmement : l'association a le droit de posséder les immeubles nécessaires à l'établissement de son siège et les sièges de ses filiales ou d'un local destiné aux réunions de ses membres ou à la réalisation de ses objectifs conformément à la loi.
Quatrièmement : l'association a le droit de céder conformément à la loi, tout immeuble qui n'est plus nécessaire à la réalisation de ses objectifs. Le produit de la cession de l'immeuble constitue une ressource pour l'association.

Art. 38 – Premièrement : toutes les transactions financières de recette ou de dépense de l'association, sont effectuées par virements ou chèques bancaires ou postaux si leur valeur dépasse cinq cents (500) dinars. La fragmentation des recettes ou des dépenses dans le but d'éviter le dépassement de la valeur sus-indiquée, n'est pas permise.
Deuxièmement : les comptes bancaires ou postaux des associations ne peuvent être gelés que par décision judiciaire.""",

    # Section 7: Registres comptables et contrôle
    """Chapitre VII
Registres et vérification des comptes

Art. 39 – Premièrement : l'association tient une comptabilité conformément au système comptable des entreprises prévu par la loi n° 96-112 du 30 décembre 1996 relative au système comptable des entreprises.
Deuxièmement : les normes comptables spécifiques aux associations sont fixées par arrêté du ministre des finances.

Art. 40 - L'association et ses filiales tiennent également les registres suivants :
Premièrement : Un registre des membres dans lequel sont consignés les noms des membres de l'association, leurs adresses, leurs nationalités, leurs âges et leurs professions.
Deuxièmement : Un registre des délibérations des organes de direction de l'association.
Troisièmement : Un registre des activités et des projets, dans lequel est consignée la nature de l'activité ou du projet.
Quatrièmement : Un registre des aides, dons, donations et legs en distinguant ceux qui sont en nature de ceux en numéraire, ceux qui sont d'origine publique de ceux d'origine privée et ceux d'origine nationale de ceux d'origine étrangère.

Art. 41 - L'association publie les données concernant les aides, dons, et donations d'origine étrangère et indique leur source, leur valeur et leur objet dans l'un des médias écrits et sur le site électronique de l'association s'il en existe et ce, dans un délai d'un mois à compter de la date de la décision de leur sollicitation ou de leur réception. Elle en informe le secrétaire général du gouvernement par lettre recommandée avec accusé de réception dans le même délai.

Art. 42 - L'association conserve ses documents et ses registres financiers pour une période de dix (10) ans.

Art. 43 – Premièrement : toute association dont les ressources annuelles dépassent cent mille (100.000) dinars, doit désigner un commissaire aux comptes choisi parmi les experts comptables inscrits au tableau de l'ordre des experts comptables de Tunisie ou inscrits au tableau de la compagnie des comptables de Tunisie à la sous-section des « techniciens en comptabilité ».
Deuxièmement : toute association dont les ressources annuelles dépassent un million (1.000.000) de dinars doit désigner un ou plusieurs commissaires aux comptes parmi ceux qui sont inscrits au tableau de l'ordre des experts comptables de Tunisie.

Art. 44 - Toute association bénéficiant du financement public présente à la cour des comptes un rapport annuel comprenant un descriptif détaillé de ses sources de financement et de ses dépenses.""",

    # Section 8: Sanctions et dispositions finales
    """Chapitre VIII
Les sanctions

Art. 45 - Pour toute infraction aux dispositions des articles 3, 4, 8 deuxièmement, 9, 10 deuxièmement, 16, 17, 18, 19, 27, 33 deuxièmement et quatrièmement, 35, 37 premièrement, 38 premièrement, 39 premièrement, 40 quatrièmement, 41, 42, 43 et 44, l'association encourt des sanctions conformément aux procédures suivantes :
Premièrement : La mise en demeure :
Le secrétaire général du gouvernement établit l'infraction commise et met en demeure l'association sur la nécessité d'y remédier dans un délai ne dépassant pas trente (30) jours à compter de la date de notification de la mise en demeure.
Deuxièmement : La suspension d'activité de l'association:
Si l'infraction n'a pas cessé dans le délai mentionné au premier paragraphe du présent article, le président du tribunal de première instance de Tunis, décide par ordonnance sur requête présentée par le secrétaire général du gouvernement, la suspension des activités de l'association pour une durée ne dépassant pas trente (30) jours. L'association peut intenter un recours contre la décision de suspension d'activité conformément aux procédures de référé.
Troisièmement : La dissolution :
Elle est prononcée par un jugement du tribunal de première instance de Tunis à la demande du secrétaire général du gouvernement ou de quiconque ayant intérêt et ce, au cas où l'association n'a pas cessé l'infraction malgré sa mise en demeure, la suspension de son activité et l'épuisement des voies de recours contre la décision de suspension d'activité.

Chapitre IX
Dispositions transitoires et finales

Art. 46 - Sont abrogées, la loi n° 59-154 du 7 novembre 1959, relative aux associations et la loi organique n° 93-80 du 26 juillet 1993 relative à l'installation des organisations non gouvernementales en Tunisie.

Art. 47 - Les dispositions du présent décret-loi ne sont pas applicables aux associations soumises à des régimes juridiques particuliers.

Art. 48 - Les dispositions du deuxième chapitre du présent décret-loi relatives à la constitution ne sont pas applicables aux associations et organisations non gouvernementales légalement établies en Tunisie à la date d'entrée en vigueur du présent décret-loi. Cependant, elles doivent se conformer aux dispositions du présent décret-loi, à l'exception des dispositions relatives à la constitution, dans le délai d'une année à compter de la date d'entrée en vigueur du présent décret-loi.

Art. 49 - Le présent décret-loi sera publié au Journal Officiel de la République Tunisienne et entre en vigueur à compter de la date de sa publication.

Tunis, le 24 septembre 2011.
Le Président de la République par intérim
Fouad Mebazaâ"""
]

# Process the document
combined_text = "\n\n".join(full_text)
chunks_count = create_fixed_chunks(combined_text)

# Initialize TF-IDF index
print("Initializing TF-IDF index...")
from chatbot.utils import setup_tfidf_index

setup_tfidf_index()
print("TF-IDF index initialized")

# Test a query
print("\nTesting a query...")
from chatbot.utils import find_relevant_chunks

test_query = "Comment créer une association en Tunisie?"
chunks = find_relevant_chunks(test_query, top_k=3)
print(f"Test query: {test_query}")
print(f"Found {len(chunks)} relevant chunks")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk.content[:150] + "...")

print("\nInitialization complete!")